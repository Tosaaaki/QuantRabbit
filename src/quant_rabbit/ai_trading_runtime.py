from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol

from quant_rabbit.decision_adjudication import adjudicate_decisions
from quant_rabbit.entry_decision import (
    EntryDecisionError,
    build_entry_decision,
    validate_sizing_receipt,
)
from quant_rabbit.exit_decision import ExitDecision, ExitDecisionError
from quant_rabbit.policy_snapshot import (
    PolicyBinding,
    PolicySnapshotError,
    load_and_verify_policy_snapshot,
)


SCHEMA_VERSION = 2
TRADE_ACTIONS = frozenset({"ENTER", "WAIT", "REQUEST_EVIDENCE", "EXIT"})
SIDES = frozenset({"LONG", "SHORT"})
ORDER_TYPES = frozenset({"MARKET", "LIMIT", "STOP", "STOP-ENTRY"})
REVIEW_POSTURES = frozenset({"NORMAL", "CAUTIOUS", "PAUSED"})


class AIRuntimeError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PreparedRun:
    manifest_path: Path
    candidate_path: Path
    run_id: str
    profile: str
    kind: str
    ready: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class AcceptedRun:
    receipt_path: Path
    run_id: str
    profile: str
    kind: str
    status: str


class DecisionSink(Protocol):
    def persist(
        self,
        *,
        state_root: Path,
        profile: str,
        receipt: Mapping[str, Any],
        repo_root: Path,
    ) -> Mapping[str, Any]: ...


class PaperLedgerSink:
    def persist(
        self,
        *,
        state_root: Path,
        profile: str,
        receipt: Mapping[str, Any],
        repo_root: Path,
    ) -> Mapping[str, Any]:
        del profile, repo_root
        execution = {
            "sink": "paper_ledger",
            "broker_mutation_allowed": False,
            "broker_order_posts": 0,
            "sent": False,
        }
        _append_jsonl_once(
            state_root / "decisions.jsonl",
            {**dict(receipt), "execution": execution},
        )
        return execution


class ReviewOverlaySink:
    def persist(
        self,
        *,
        state_root: Path,
        profile: str,
        receipt: Mapping[str, Any],
        repo_root: Path,
    ) -> Mapping[str, Any]:
        del profile
        execution = {
            "sink": "review_overlay",
            "broker_mutation_allowed": False,
            "broker_order_posts": 0,
            "sent": False,
        }
        persisted = {**dict(receipt), "execution": execution}
        _append_jsonl_once(state_root / "reviews.jsonl", persisted)
        target = state_root / "strategic_review.json"
        _atomic_write_json(target, persisted)
        return execution


class LiveGatewaySink:
    def persist(
        self,
        *,
        state_root: Path,
        profile: str,
        receipt: Mapping[str, Any],
        repo_root: Path,
    ) -> Mapping[str, Any]:
        del profile
        from quant_rabbit.ai_live_gateway import (
            AILiveGatewayError,
            execute_ai_trade_candidate,
        )

        try:
            result = execute_ai_trade_candidate(
                repo_root=repo_root,
                state_root=state_root,
                receipt=receipt,
            )
        except AILiveGatewayError as exc:
            raise AIRuntimeError("LIVE_GATEWAY_REJECTED", str(exc)) from exc
        _append_jsonl_once(
            state_root / "decisions.jsonl",
            {**dict(receipt), "execution": dict(result)},
        )
        return result


SINKS: dict[str, DecisionSink] = {
    "paper_ledger": PaperLedgerSink(),
    "review_overlay": ReviewOverlaySink(),
    "live_gateway": LiveGatewaySink(),
}


def prepare_run(
    *,
    config_path: Path,
    profile: str,
    repo_root: Path,
    state_root: Path | None = None,
    now: datetime | None = None,
) -> PreparedRun:
    current = _utc_now(now)
    config = _load_object(config_path, "runtime config")
    profile_config = _profile_config(config, profile)
    resolved_state_root = state_root or _state_root(config)
    descriptors: list[dict[str, Any]] = []
    blockers: list[str] = []

    workers = profile_config.get("workers")
    if not isinstance(workers, Mapping) or not workers:
        raise AIRuntimeError("PROFILE_INVALID", f"profile {profile!r} has no workers")
    for worker_name, sources in workers.items():
        if not isinstance(worker_name, str) or not worker_name.strip() or not isinstance(sources, list):
            raise AIRuntimeError("PROFILE_INVALID", "workers must map names to source lists")
        for source in sources:
            descriptor = _describe_source(
                worker=worker_name,
                source=source,
                repo_root=repo_root,
                state_root=resolved_state_root,
                now=current,
            )
            descriptors.append(descriptor)
            if descriptor["required"] and descriptor["status"] != "READY":
                blockers.append(f"{worker_name}:{descriptor['path']}:{descriptor['status']}")

    source_digest = _sha256_json(_source_material(descriptors))
    run_id = f"{profile}-{current.strftime('%Y%m%dT%H%M%S%fZ')}-{source_digest[:12]}"
    run_dir = resolved_state_root / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    candidate_path = run_dir / "candidate.json"
    receipt_path = run_dir / "receipt.json"
    kind = str(profile_config.get("kind") or "").strip().lower()
    if kind not in {"trade", "review"}:
        raise AIRuntimeError("PROFILE_INVALID", f"unsupported profile kind: {kind!r}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "profile": profile,
        "kind": kind,
        "prepared_at_utc": current.isoformat(),
        "decision_max_age_seconds": _positive_int(
            profile_config.get("decision_max_age_seconds"),
            "decision_max_age_seconds",
        ),
        "sink": str(profile_config.get("sink") or ""),
        "ready": not blockers,
        "blockers": blockers,
        "source_digest": source_digest,
        "sources": descriptors,
        "candidate_path": str(candidate_path),
        "receipt_path": str(receipt_path),
        "execution": {
            "mode": (
                "live"
                if profile_config.get("sink") == "live_gateway"
                else "paper"
                if kind == "trade"
                else "review"
            ),
            "broker_mutation_allowed": profile_config.get("sink") == "live_gateway",
            "broker_api_calls_allowed": profile_config.get("sink") == "live_gateway",
        },
        "candidate_schema": _candidate_schema(
            kind,
            allowed_actions=_allowed_trade_actions(profile_config, kind=kind),
        ),
    }
    _atomic_write_json(manifest_path, manifest)
    _atomic_write_json(candidate_path, _candidate_template(manifest))
    return PreparedRun(
        manifest_path=manifest_path,
        candidate_path=candidate_path,
        run_id=run_id,
        profile=profile,
        kind=kind,
        ready=not blockers,
        blockers=tuple(blockers),
    )


def accept_run(
    *,
    config_path: Path,
    manifest_path: Path,
    candidate_path: Path,
    repo_root: Path,
    state_root: Path | None = None,
    now: datetime | None = None,
) -> AcceptedRun:
    current = _utc_now(now)
    config = _load_object(config_path, "runtime config")
    manifest = _load_object(manifest_path, "run manifest")
    candidate = _load_object(candidate_path, "AI candidate")
    profile = _required_text(manifest, "profile")
    profile_config = _profile_config(config, profile)
    resolved_state_root = state_root or _state_root(config)
    _validate_manifest(
        manifest,
        profile_config,
        repo_root=repo_root,
        state_root=resolved_state_root,
        manifest_path=manifest_path,
        candidate_path=candidate_path,
        now=current,
    )
    _validate_candidate(candidate, manifest=manifest, now=current)

    decision_contracts: dict[str, Any] = {}
    if manifest.get("kind") == "trade":
        evidence_packet = _load_bound_evidence_packet(
            manifest,
            repo_root=repo_root,
            state_root=resolved_state_root,
        )
        decision_contracts = _build_decision_contracts(
            candidate,
            manifest=manifest,
            profile_config=profile_config,
            evidence_packet=evidence_packet,
            now=current,
        )
        if (
            str(profile_config.get("sink") or "") == "live_gateway"
            and str(candidate.get("action") or "").upper() in {"ENTER", "EXIT"}
        ):
            decision_contracts["hotpath_lease"] = _require_active_hotpath_lease(
                resolved_state_root,
                run_id=str(manifest["run_id"]),
                now=current,
            )
            decision_contracts["policy_snapshot"] = _load_bound_live_policy_snapshot(
                manifest,
                repo_root=repo_root,
                state_root=resolved_state_root,
                now=current,
            )

    kind = _required_text(manifest, "kind")
    sink_name = _required_text(manifest, "sink")
    sink = SINKS.get(sink_name)
    if sink is None:
        raise AIRuntimeError("SINK_UNAVAILABLE", f"unknown decision sink: {sink_name}")
    status = (
        "ACCEPTED_LIVE"
        if sink_name == "live_gateway"
        else "ACCEPTED_PAPER"
        if kind == "trade"
        else "ACCEPTED_REVIEW"
    )
    candidate_sha256 = _sha256_json(candidate)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "profile": profile,
        "kind": kind,
        "accepted_at_utc": current.isoformat(),
        "status": status,
        "model": candidate["model"],
        "reasoning_effort": candidate["reasoning_effort"],
        "source_digest": manifest["source_digest"],
        "candidate_sha256": candidate_sha256,
        "decision": candidate,
        **decision_contracts,
        "execution": {"sink": sink_name},
    }
    receipt_path = Path(_required_text(manifest, "receipt_path"))
    if receipt_path.exists():
        existing = _load_object(receipt_path, "existing receipt")
        if existing.get("candidate_sha256") != candidate_sha256:
            raise AIRuntimeError("RUN_ALREADY_ACCEPTED", "run already has a different accepted candidate")
        _finish_hotpath_lease_if_owned(
            resolved_state_root,
            run_id=str(manifest["run_id"]),
            status=str(existing.get("status") or status),
            now=current,
        )
        return AcceptedRun(
            receipt_path=receipt_path,
            run_id=str(manifest["run_id"]),
            profile=profile,
            kind=kind,
            status=str(existing.get("status") or status),
        )
    execution = sink.persist(
        state_root=resolved_state_root,
        profile=profile,
        receipt=receipt,
        repo_root=repo_root,
    )
    receipt["execution"] = dict(execution)
    if sink_name == "live_gateway":
        if execution.get("sent") is True:
            status = "ACCEPTED_LIVE_SENT"
        elif execution.get("status") == "NO_BROKER_ACTION":
            status = "ACCEPTED_NO_BROKER_ACTION"
        else:
            status = "ACCEPTED_LIVE_BLOCKED"
        receipt["status"] = status
    _atomic_write_json(receipt_path, receipt)
    _finish_hotpath_lease_if_owned(
        resolved_state_root,
        run_id=str(manifest["run_id"]),
        status=status,
        now=current,
    )
    return AcceptedRun(
        receipt_path=receipt_path,
        run_id=str(manifest["run_id"]),
        profile=profile,
        kind=kind,
        status=status,
    )


def _validate_manifest(
    manifest: Mapping[str, Any],
    profile_config: Mapping[str, Any],
    *,
    repo_root: Path,
    state_root: Path,
    manifest_path: Path,
    candidate_path: Path,
    now: datetime,
) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise AIRuntimeError("MANIFEST_INVALID", "unsupported manifest schema")
    run_id = _required_text(manifest, "run_id")
    expected_run_dir = (state_root / "runs" / run_id).resolve()
    if manifest_path.resolve() != expected_run_dir / "manifest.json":
        raise AIRuntimeError("MANIFEST_SCOPE_MISMATCH", "manifest path is outside its configured run directory")
    if candidate_path.resolve() != expected_run_dir / "candidate.json":
        raise AIRuntimeError("MANIFEST_SCOPE_MISMATCH", "candidate path is outside its configured run directory")
    if Path(_required_text(manifest, "candidate_path")).resolve() != candidate_path.resolve():
        raise AIRuntimeError("MANIFEST_SCOPE_MISMATCH", "manifest candidate_path does not match the supplied candidate")
    if Path(_required_text(manifest, "receipt_path")).resolve() != expected_run_dir / "receipt.json":
        raise AIRuntimeError("MANIFEST_SCOPE_MISMATCH", "receipt path is outside its configured run directory")
    expected_kind = str(profile_config.get("kind") or "").strip().lower()
    expected_sink = str(profile_config.get("sink") or "").strip()
    if manifest.get("kind") != expected_kind or manifest.get("sink") != expected_sink:
        raise AIRuntimeError("MANIFEST_SCOPE_MISMATCH", "manifest kind or sink differs from profile config")
    if manifest.get("ready") is not True:
        raise AIRuntimeError(
            "EVIDENCE_NOT_READY",
            "required evidence was missing or stale: " + ", ".join(manifest.get("blockers") or []),
        )
    prepared_at = _parse_utc(manifest.get("prepared_at_utc"), "prepared_at_utc")
    max_age = _positive_int(profile_config.get("decision_max_age_seconds"), "decision_max_age_seconds")
    age = (now - prepared_at).total_seconds()
    if age < -60 or age > max_age:
        raise AIRuntimeError("MANIFEST_STALE", f"manifest age {age:.1f}s exceeds {max_age}s")
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise AIRuntimeError("MANIFEST_INVALID", "sources must be a list")
    refreshed: list[dict[str, Any]] = []
    workers = profile_config.get("workers")
    if not isinstance(workers, Mapping):
        raise AIRuntimeError("PROFILE_INVALID", "profile workers must be an object")
    for worker, configured_sources in workers.items():
        if not isinstance(worker, str) or not isinstance(configured_sources, list):
            raise AIRuntimeError("PROFILE_INVALID", "profile workers are malformed")
        for source in configured_sources:
            refreshed.append(_describe_source(
                worker=worker,
                source=source,
                repo_root=repo_root,
                state_root=state_root,
                now=now,
            ))
    if len(refreshed) != len(sources) or _sha256_json(_source_material(refreshed)) != manifest.get("source_digest"):
        raise AIRuntimeError("EVIDENCE_CHANGED", "input evidence changed after the run was prepared")
    configured_scope = [
        {key: row.get(key) for key in ("worker", "path", "required", "max_age_seconds")}
        for row in refreshed
    ]
    manifest_scope = [
        {key: row.get(key) for key in ("worker", "path", "required", "max_age_seconds")}
        for row in sources
        if isinstance(row, Mapping)
    ]
    if manifest_scope != configured_scope:
        raise AIRuntimeError("MANIFEST_SCOPE_MISMATCH", "manifest sources differ from profile config")


def _validate_candidate(candidate: Mapping[str, Any], *, manifest: Mapping[str, Any], now: datetime) -> None:
    if candidate.get("schema_version") != SCHEMA_VERSION:
        raise AIRuntimeError("CANDIDATE_INVALID", "unsupported candidate schema")
    for field in ("run_id", "profile", "kind"):
        if candidate.get(field) != manifest.get(field):
            raise AIRuntimeError("CANDIDATE_SCOPE_MISMATCH", f"candidate {field} does not match manifest")
    _required_text(candidate, "model")
    _required_text(candidate, "reasoning_effort")
    decided_at = _parse_utc(candidate.get("decided_at_utc"), "decided_at_utc")
    prepared_at = _parse_utc(manifest.get("prepared_at_utc"), "prepared_at_utc")
    if decided_at < prepared_at or decided_at > now.replace(microsecond=999999):
        raise AIRuntimeError("CANDIDATE_TIMESTAMP_INVALID", "decision must be authored after prepare and not in the future")
    if candidate.get("source_digest") != manifest.get("source_digest"):
        raise AIRuntimeError("CANDIDATE_SCOPE_MISMATCH", "candidate source_digest does not match manifest")
    _required_text(candidate, "thesis")
    evidence_refs = candidate.get("evidence_refs")
    if not isinstance(evidence_refs, list) or not evidence_refs or any(not isinstance(v, str) or not v.strip() for v in evidence_refs):
        raise AIRuntimeError("CANDIDATE_INVALID", "evidence_refs must contain at least one source reference")
    allowed_refs = {
        f"{item.get('worker')}:{item.get('path')}"
        for item in manifest.get("sources") or []
        if isinstance(item, Mapping)
    }
    unknown_refs = sorted(set(evidence_refs) - allowed_refs)
    if unknown_refs:
        raise AIRuntimeError("CANDIDATE_INVALID", "unknown evidence_refs: " + ", ".join(unknown_refs))
    if manifest.get("kind") == "trade":
        schema = manifest.get("candidate_schema")
        allowed_actions = schema.get("actions") if isinstance(schema, Mapping) else None
        action = str(candidate.get("action") or "").strip().upper()
        if not isinstance(allowed_actions, list) or action not in allowed_actions:
            raise AIRuntimeError(
                "ACTION_NOT_ENABLED",
                f"action {action or 'UNKNOWN'} is not enabled for this profile",
            )
        _validate_trade_candidate(candidate)
    elif manifest.get("kind") == "review":
        _validate_review_candidate(candidate, now=now)
    else:
        raise AIRuntimeError("CANDIDATE_INVALID", "unsupported candidate kind")


def _load_bound_evidence_packet(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
    state_root: Path,
) -> dict[str, Any]:
    descriptors = [
        item
        for item in manifest.get("sources") or []
        if isinstance(item, Mapping) and item.get("worker") == "evidence"
    ]
    if len(descriptors) != 1:
        raise AIRuntimeError("EVIDENCE_PACKET_INVALID", "trade profile requires exactly one evidence packet")
    descriptor = descriptors[0]
    raw_path = _required_text(descriptor, "path")
    path = (
        state_root / raw_path.removeprefix("@state/")
        if raw_path.startswith("@state/")
        else Path(raw_path).expanduser()
    )
    if not path.is_absolute():
        path = repo_root / path
    packet = _load_object(path, "AI evidence packet")
    from quant_rabbit.ai_evidence_adapter import CONTRACT as EVIDENCE_CONTRACT

    if packet.get("contract") != EVIDENCE_CONTRACT or packet.get("schema_version") != 1:
        raise AIRuntimeError("EVIDENCE_PACKET_INVALID", "unsupported evidence packet contract")
    supplied = packet.get("packet_sha256")
    body = {key: value for key, value in packet.items() if key != "packet_sha256"}
    if not isinstance(supplied, str) or supplied != _sha256_json(body):
        raise AIRuntimeError("EVIDENCE_PACKET_TAMPERED", "evidence packet content digest does not match")
    if descriptor.get("sha256") != _sha256_file(path):
        raise AIRuntimeError("EVIDENCE_CHANGED", "evidence packet bytes differ from the prepared manifest")
    return packet


def _load_bound_live_policy_snapshot(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
    state_root: Path,
    now: datetime,
) -> dict[str, Any]:
    descriptors = [
        item
        for item in manifest.get("sources") or []
        if isinstance(item, Mapping) and item.get("worker") == "policy"
    ]
    if len(descriptors) != 1:
        raise AIRuntimeError(
            "POLICY_SNAPSHOT_INVALID",
            "live broker mutation requires exactly one sealed policy source",
        )
    descriptor = descriptors[0]
    raw_path = _required_text(descriptor, "path")
    path = (
        state_root / raw_path.removeprefix("@state/")
        if raw_path.startswith("@state/")
        else Path(raw_path).expanduser()
    )
    if not path.is_absolute():
        path = repo_root / path
    environment = {
        "project_key": os.environ.get("QR_AI_PROJECT_KEY"),
        "broker_account_id": os.environ.get("QR_AI_BROKER_ACCOUNT_ID"),
        "environment": os.environ.get("QR_AI_ENVIRONMENT"),
        "revocation_epoch": os.environ.get("QR_AI_POLICY_REVOCATION_EPOCH"),
    }
    missing = [name for name, value in environment.items() if value is None or not str(value).strip()]
    if missing:
        raise AIRuntimeError(
            "POLICY_BINDING_MISSING",
            "live policy binding environment is incomplete: " + ", ".join(missing),
        )
    required_pages = tuple(
        value.strip()
        for value in os.environ.get("QR_AI_REQUIRED_POLICY_SOURCE_PAGES", "").split(",")
        if value.strip()
    )
    if not required_pages:
        raise AIRuntimeError(
            "POLICY_SOURCE_BINDING_MISSING",
            "live broker mutation requires explicit policy source-page bindings",
        )
    try:
        revocation_epoch = int(str(environment["revocation_epoch"]))
        return load_and_verify_policy_snapshot(
            path,
            binding=PolicyBinding(
                project_key=str(environment["project_key"]).strip(),
                broker_account_id=str(environment["broker_account_id"]).strip(),
                environment=str(environment["environment"]).strip(),
                revocation_epoch=revocation_epoch,
            ),
            now=now,
            required_source_pages=required_pages,
        )
    except (ValueError, PolicySnapshotError) as exc:
        raise AIRuntimeError(
            "POLICY_SNAPSHOT_REJECTED",
            f"{getattr(exc, 'code', 'POLICY_BINDING_INVALID')}: {exc}",
        ) from exc


def _require_active_hotpath_lease(
    state_root: Path,
    *,
    run_id: str,
    now: datetime,
) -> dict[str, Any]:
    path = state_root / "hotpath_lease.json"
    if path.is_symlink() or not path.is_file():
        raise AIRuntimeError("HOTPATH_LEASE_MISSING", "live mutation requires an active hot-path lease")
    lease = _load_object(path, "hot-path lease")
    supplied = lease.get("lease_sha256")
    material = {key: value for key, value in lease.items() if key != "lease_sha256"}
    if not isinstance(supplied, str) or supplied != _sha256_json(material):
        raise AIRuntimeError("HOTPATH_LEASE_TAMPERED", "hot-path lease seal does not match")
    if lease.get("status") != "ACTIVE" or lease.get("run_id") != run_id:
        raise AIRuntimeError("HOTPATH_LEASE_MISMATCH", "another run owns the hot-path lease")
    expires = _parse_utc(lease.get("expires_at_utc"), "hotpath_lease.expires_at_utc")
    if now >= expires:
        raise AIRuntimeError("HOTPATH_LEASE_EXPIRED", "hot-path lease expired before acceptance")
    return lease


def _finish_hotpath_lease_if_owned(
    state_root: Path,
    *,
    run_id: str,
    status: str,
    now: datetime,
) -> None:
    path = state_root / "hotpath_lease.json"
    if not path.is_file() or path.is_symlink():
        return
    lease = _load_object(path, "hot-path lease")
    material = {key: value for key, value in lease.items() if key != "lease_sha256"}
    if lease.get("lease_sha256") != _sha256_json(material):
        raise AIRuntimeError("HOTPATH_LEASE_TAMPERED", "hot-path lease seal does not match")
    if lease.get("status") != "ACTIVE" or lease.get("run_id") != run_id:
        return
    terminal = {
        **material,
        "status": "TERMINAL",
        "terminal_status": status,
        "completed_at_utc": now.isoformat(),
    }
    _atomic_write_json(path, {**terminal, "lease_sha256": _sha256_json(terminal)})


def _build_decision_contracts(
    candidate: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    profile_config: Mapping[str, Any],
    evidence_packet: Mapping[str, Any],
    now: datetime,
) -> dict[str, Any]:
    action = str(candidate["action"]).upper()
    # The receipt retains the full AI thesis.  The typed decision stores a
    # bounded, content-addressed reference so a detailed thesis cannot exceed
    # the decision contract's per-reason limit.
    candidate_reason = f"ai_candidate_sha256:{_sha256_json(candidate)}"
    if action in {"ENTER", "EXIT"} and evidence_packet.get("status") != "READY":
        raise AIRuntimeError("EVIDENCE_NOT_READY", "broker mutation requires a READY evidence packet")
    broker_epoch = evidence_packet.get("broker_epoch")
    if not isinstance(broker_epoch, Mapping):
        raise AIRuntimeError("EVIDENCE_PACKET_INVALID", "evidence packet has no broker epoch")
    last_transaction_id = broker_epoch.get("last_transaction_id")
    if not isinstance(last_transaction_id, str) or not last_transaction_id:
        raise AIRuntimeError("EVIDENCE_PACKET_INVALID", "broker epoch has no last transaction id")
    observed_at = _parse_utc(broker_epoch.get("as_of_utc"), "broker_epoch.as_of_utc")
    decided_at = _parse_utc(candidate.get("decided_at_utc"), "decided_at_utc")
    ttl_seconds = min(
        _positive_int(profile_config.get("decision_max_age_seconds"), "decision_max_age_seconds"),
        600,
    )
    entry: dict[str, Any] | None = None
    exit_decision: dict[str, Any] | None = None
    if action == "ENTER":
        proposal = dict(candidate["orders"][0])
        expected_binding = {
            "packet_sha256": evidence_packet.get("packet_sha256"),
            "source_set_sha256": evidence_packet.get("source_set_sha256"),
            "broker_epoch": last_transaction_id,
        }
        if proposal.get("evidence_binding") != expected_binding:
            raise AIRuntimeError("ORDER_INVALID", "order evidence_binding does not match the sealed packet")
        entry = build_entry_decision(
            action="ENTER",
            cycle_id=str(manifest["run_id"]),
            broker_epoch=last_transaction_id,
            evidence_observed_at_utc=observed_at,
            proposal=proposal,
            reasons=(candidate_reason,),
            ttl_seconds=ttl_seconds,
            created_at_utc=decided_at,
        )
    elif action in {"WAIT", "REQUEST_EVIDENCE"}:
        entry = build_entry_decision(
            action=action,
            cycle_id=str(manifest["run_id"]),
            broker_epoch=last_transaction_id,
            evidence_observed_at_utc=observed_at,
            requested_evidence=tuple(candidate.get("requested_evidence") or ()),
            reasons=(candidate_reason,),
            ttl_seconds=ttl_seconds,
            created_at_utc=decided_at,
        )
    else:
        item = candidate["position_actions"][0]
        try:
            exit_decision = ExitDecision.create(
                action=item["action"],
                cycle_id=str(manifest["run_id"]),
                broker_epoch=last_transaction_id,
                position_revision=item["position_revision"],
                trade_id=item["trade_id"],
                instrument=item["instrument"],
                owner_binding=item["owner_binding"],
                created_at_utc=decided_at,
                ttl_seconds=ttl_seconds,
                emergency_eligible=item.get("emergency_eligible", False),
                units=item.get("units"),
                stop_loss=item.get("stop_loss"),
                take_profit=item.get("take_profit"),
                reason=item["reason"],
                evidence_refs=tuple(item.get("evidence_refs") or candidate.get("evidence_refs") or ()),
            ).to_dict()
        except ExitDecisionError as exc:
            raise AIRuntimeError("EXIT_DECISION_INVALID", f"{exc.code}: {exc}") from exc
    try:
        adjudication = adjudicate_decisions(
            cycle_id=str(manifest["run_id"]),
            broker_epoch=last_transaction_id,
            entry_proposals=(() if entry is None else (entry,)),
            exit_proposals=(() if exit_decision is None else (exit_decision,)),
            now=now,
        )
    except ValueError as exc:
        raise AIRuntimeError("ADJUDICATION_REJECTED", str(exc)) from exc
    return {
        "evidence_packet": dict(evidence_packet),
        "entry_decision": entry,
        "exit_decisions": [] if exit_decision is None else [exit_decision],
        "adjudication": adjudication,
    }


def _validate_trade_candidate(candidate: Mapping[str, Any]) -> None:
    action = _required_text(candidate, "action").upper()
    if action not in TRADE_ACTIONS:
        raise AIRuntimeError("CANDIDATE_INVALID", f"unsupported action: {action}")
    confidence = candidate.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
        raise AIRuntimeError("CANDIDATE_INVALID", "confidence must be within 0..1")
    orders = candidate.get("orders")
    position_actions = candidate.get("position_actions")
    requested = candidate.get("requested_evidence")
    if not isinstance(orders, list) or not isinstance(position_actions, list) or not isinstance(requested, list):
        raise AIRuntimeError("CANDIDATE_INVALID", "orders, position_actions, and requested_evidence must be lists")
    if action == "ENTER":
        if not orders or position_actions or requested:
            raise AIRuntimeError("CANDIDATE_INVALID", "ENTER requires orders only")
        if len(orders) != 1:
            raise AIRuntimeError("CANDIDATE_INVALID", "ENTER requires exactly one order")
        for order in orders:
            _validate_order(order)
    elif action == "EXIT":
        if orders or len(position_actions) != 1 or requested:
            raise AIRuntimeError("CANDIDATE_INVALID", "EXIT requires exactly one position action")
        item = position_actions[0]
        if not isinstance(item, Mapping):
            raise AIRuntimeError("CANDIDATE_INVALID", "position action must be an object")
        for field in ("action", "trade_id", "instrument", "position_revision", "reason"):
            _required_text(item, field)
        _required_text(item, "reason")
        owner = item.get("owner_binding")
        if not isinstance(owner, Mapping) or str(owner.get("owner_kind") or "").upper() != "AI_SYSTEM":
            raise AIRuntimeError("CANDIDATE_INVALID", "EXIT is limited to exact AI_SYSTEM ownership")
    elif action == "REQUEST_EVIDENCE":
        if orders or position_actions or not requested or any(not isinstance(v, str) or not v.strip() for v in requested):
            raise AIRuntimeError("CANDIDATE_INVALID", "REQUEST_EVIDENCE requires requested_evidence only")
    elif orders or position_actions or requested:
        raise AIRuntimeError("CANDIDATE_INVALID", "WAIT cannot include orders, position actions, or requested evidence")


def _validate_order(order: Any) -> None:
    if not isinstance(order, Mapping):
        raise AIRuntimeError("ORDER_INVALID", "order must be an object")
    for field in (
        "decision_id",
        "pair",
        "side",
        "method",
        "vehicle",
        "order_type",
        "rationale",
    ):
        _required_text(order, field)
    pair = str(order["pair"]).upper()
    if len(pair) != 7 or pair[3] != "_" or not pair.replace("_", "").isalpha():
        raise AIRuntimeError("ORDER_INVALID", f"invalid pair: {pair}")
    side = str(order["side"]).upper()
    if side not in SIDES:
        raise AIRuntimeError("ORDER_INVALID", f"invalid side: {side}")
    order_type = str(order["order_type"]).upper()
    if order_type not in ORDER_TYPES:
        raise AIRuntimeError("ORDER_INVALID", f"invalid order_type: {order_type}")
    from quant_rabbit.models import TradeMethod

    try:
        TradeMethod.parse(str(order["method"]))
    except ValueError as exc:
        raise AIRuntimeError("ORDER_INVALID", str(exc)) from exc
    vehicle = str(order["vehicle"]).upper()
    normalized_vehicle = "STOP" if order_type in {"STOP", "STOP-ENTRY"} else order_type
    if vehicle != normalized_vehicle:
        raise AIRuntimeError("ORDER_INVALID", "vehicle must match order_type")
    entry = _positive_number(order.get("entry"), "entry")
    tp = _positive_number(order.get("take_profit"), "take_profit")
    sl = _positive_number(order.get("stop_loss"), "stop_loss")
    units = order.get("units")
    if isinstance(units, bool) or not isinstance(units, int) or units <= 0:
        raise AIRuntimeError("ORDER_INVALID", "units must be a positive integer")
    if "allocation_multiplier" in order:
        raise AIRuntimeError("ORDER_INVALID", "allocation_multiplier is not part of dynamic AI sizing")
    try:
        validate_sizing_receipt(order.get("sizing_receipt"))
    except EntryDecisionError as exc:
        raise AIRuntimeError("ORDER_INVALID", f"invalid sizing_receipt: {exc}") from exc
    if order["sizing_receipt"].get("final_units") != units:
        raise AIRuntimeError("ORDER_INVALID", "units must equal sizing_receipt.final_units")
    for proof_name in ("evidence_binding", "net_edge_proof", "cost_proof"):
        if not isinstance(order.get(proof_name), Mapping):
            raise AIRuntimeError("ORDER_INVALID", f"{proof_name} must be an object")
    if side == "LONG" and not sl < entry < tp:
        raise AIRuntimeError("ORDER_GEOMETRY_INVALID", "LONG requires stop_loss < entry < take_profit")
    if side == "SHORT" and not tp < entry < sl:
        raise AIRuntimeError("ORDER_GEOMETRY_INVALID", "SHORT requires take_profit < entry < stop_loss")


def _validate_review_candidate(candidate: Mapping[str, Any], *, now: datetime) -> None:
    _required_text(candidate, "regime")
    posture = _required_text(candidate, "risk_posture").upper()
    if posture not in REVIEW_POSTURES:
        raise AIRuntimeError("CANDIDATE_INVALID", f"unsupported risk_posture: {posture}")
    valid_until = _parse_utc(candidate.get("valid_until_utc"), "valid_until_utc")
    if valid_until <= now:
        raise AIRuntimeError("CANDIDATE_INVALID", "strategic review must expire in the future")
    for field in ("themes", "instructions"):
        value = candidate.get(field)
        if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
            raise AIRuntimeError("CANDIDATE_INVALID", f"{field} must be a list of non-empty strings")


def _candidate_schema(
    kind: str,
    *,
    allowed_actions: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    common = [
        "schema_version", "run_id", "profile", "kind", "model",
        "reasoning_effort", "decided_at_utc", "source_digest", "thesis",
        "evidence_refs", "extensions",
    ]
    if kind == "trade":
        return {
            "required": common + ["action", "confidence", "orders", "position_actions", "requested_evidence"],
            "actions": list(allowed_actions or sorted(TRADE_ACTIONS)),
            "order_fields": [
                "decision_id", "pair", "side", "method", "vehicle", "order_type", "entry",
                "take_profit", "stop_loss", "units", "sizing_receipt",
                "evidence_binding", "net_edge_proof", "cost_proof", "rationale", "extensions",
            ],
            "position_action_fields": [
                "action", "trade_id", "instrument", "position_revision",
                "owner_binding", "emergency_eligible", "units", "stop_loss",
                "take_profit", "reason", "evidence_refs",
            ],
        }
    return {
        "required": common + ["regime", "risk_posture", "valid_until_utc", "themes", "instructions"],
        "risk_postures": sorted(REVIEW_POSTURES),
    }


def _allowed_trade_actions(
    profile_config: Mapping[str, Any],
    *,
    kind: str,
) -> tuple[str, ...] | None:
    if kind != "trade":
        return None
    raw = profile_config.get("allowed_actions")
    if not isinstance(raw, list) or not raw:
        raise AIRuntimeError(
            "PROFILE_INVALID",
            "trade profiles must declare a non-empty allowed_actions list",
        )
    normalized: list[str] = []
    for item in raw:
        action = str(item or "").strip().upper()
        if action not in TRADE_ACTIONS or action in normalized:
            raise AIRuntimeError("PROFILE_INVALID", f"invalid allowed trade action: {item!r}")
        normalized.append(action)
    if "WAIT" not in normalized or "REQUEST_EVIDENCE" not in normalized:
        raise AIRuntimeError(
            "PROFILE_INVALID",
            "trade profiles must retain WAIT and REQUEST_EVIDENCE",
        )
    if profile_config.get("sink") == "live_gateway" and "EXIT" in normalized:
        raise AIRuntimeError(
            "PROFILE_INVALID",
            "live_gateway is entry-only until the owner/revision-bound exit sink is deployed",
        )
    return tuple(normalized)


def _candidate_template(manifest: Mapping[str, Any]) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "profile": manifest["profile"],
        "kind": manifest["kind"],
        "model": "REPLACE_WITH_RUNTIME_MODEL",
        "reasoning_effort": "REPLACE_WITH_RUNTIME_REASONING",
        "decided_at_utc": "REPLACE_WITH_CURRENT_UTC",
        "source_digest": manifest["source_digest"],
        "thesis": "REPLACE_WITH_AI_THESIS",
        "evidence_refs": [],
        "extensions": {},
    }
    if manifest["kind"] == "trade":
        base.update({
            "action": "REQUEST_EVIDENCE" if not manifest["ready"] else "WAIT",
            "confidence": 0.0,
            "orders": [],
            "position_actions": [],
            "requested_evidence": list(manifest.get("blockers") or []),
        })
    else:
        base.update({
            "regime": "UNKNOWN",
            "risk_posture": "PAUSED" if not manifest["ready"] else "CAUTIOUS",
            "valid_until_utc": "REPLACE_WITH_FUTURE_UTC",
            "themes": [],
            "instructions": [],
        })
    return base


def _describe_source(
    *,
    worker: str,
    source: Any,
    repo_root: Path,
    state_root: Path,
    now: datetime,
) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise AIRuntimeError("PROFILE_INVALID", "worker source must be an object")
    raw_path = _required_text(source, "path")
    if raw_path.startswith("@state/"):
        path = state_root / raw_path.removeprefix("@state/")
    else:
        path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    required = source.get("required") is True
    max_age = _positive_int(source.get("max_age_seconds"), f"{raw_path}.max_age_seconds")
    status = "MISSING"
    sha256: str | None = None
    size: int | None = None
    modified_at: str | None = None
    age_seconds: float | None = None
    if path.is_file():
        stat = path.stat()
        size = stat.st_size
        modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        modified_at = modified.isoformat()
        age_seconds = max(0.0, (now - modified).total_seconds())
        sha256 = _sha256_file(path)
        status = "READY" if age_seconds <= max_age else "STALE"
    return {
        "worker": worker,
        "path": raw_path,
        "required": required,
        "max_age_seconds": max_age,
        "status": status,
        "sha256": sha256,
        "size": size,
        "modified_at_utc": modified_at,
        "age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
    }


def _source_material(descriptors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in descriptor.items() if key != "age_seconds"}
        for descriptor in descriptors
    ]


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AIRuntimeError("JSON_READ_FAILED", f"unable to read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AIRuntimeError("JSON_READ_FAILED", f"{label} must be a JSON object")
    return value


def _profile_config(config: Mapping[str, Any], profile: str) -> dict[str, Any]:
    profiles = config.get("profiles")
    value = profiles.get(profile) if isinstance(profiles, Mapping) else None
    if not isinstance(value, Mapping):
        raise AIRuntimeError("PROFILE_NOT_FOUND", f"unknown profile: {profile}")
    return dict(value)


def _state_root(config: Mapping[str, Any]) -> Path:
    override = os.environ.get("QR_AI_TRADER_STATE_ROOT")
    raw = override or str(config.get("state_root") or "")
    if not raw.strip():
        raise AIRuntimeError("CONFIG_INVALID", "state_root is required")
    return Path(raw).expanduser()


def _required_text(value: Mapping[str, Any], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise AIRuntimeError("FIELD_REQUIRED", f"{field} must be a non-empty string")
    return item.strip()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AIRuntimeError("CONFIG_INVALID", f"{label} must be a positive integer")
    return value


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AIRuntimeError("ORDER_INVALID", f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise AIRuntimeError("ORDER_INVALID", f"{label} must be positive and finite")
    return parsed


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise AIRuntimeError("TIMESTAMP_INVALID", f"{label} is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AIRuntimeError("TIMESTAMP_INVALID", f"invalid {label}: {value}") from exc
    if parsed.tzinfo is None:
        raise AIRuntimeError("TIMESTAMP_INVALID", f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _utc_now(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise AIRuntimeError("TIMESTAMP_INVALID", "now must include a timezone")
    return current.astimezone(timezone.utc)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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


def _append_jsonl_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        run_id = value.get("run_id")
        if run_id is not None:
            handle.seek(0)
            for existing_line in handle:
                try:
                    existing = json.loads(existing_line)
                except json.JSONDecodeError:
                    continue
                if isinstance(existing, Mapping) and existing.get("run_id") == run_id:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    return
            handle.seek(0, os.SEEK_END)
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
