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


SCHEMA_VERSION = 1
TRADE_ACTIONS = frozenset({"TRADE", "WAIT", "REQUEST_EVIDENCE", "CLOSE"})
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
        "candidate_schema": _candidate_schema(kind),
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
        "execution": {"sink": sink_name},
    }
    receipt_path = Path(_required_text(manifest, "receipt_path"))
    if receipt_path.exists():
        existing = _load_object(receipt_path, "existing receipt")
        if existing.get("candidate_sha256") != candidate_sha256:
            raise AIRuntimeError("RUN_ALREADY_ACCEPTED", "run already has a different accepted candidate")
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
        _validate_trade_candidate(candidate)
    elif manifest.get("kind") == "review":
        _validate_review_candidate(candidate, now=now)
    else:
        raise AIRuntimeError("CANDIDATE_INVALID", "unsupported candidate kind")


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
    if action == "TRADE":
        if not orders or position_actions or requested:
            raise AIRuntimeError("CANDIDATE_INVALID", "TRADE requires orders only")
        for order in orders:
            _validate_order(order)
    elif action == "CLOSE":
        if orders or len(position_actions) != 1 or requested:
            raise AIRuntimeError("CANDIDATE_INVALID", "CLOSE requires exactly one position action")
        item = position_actions[0]
        if not isinstance(item, Mapping):
            raise AIRuntimeError("CANDIDATE_INVALID", "position action must be an object")
        _required_text(item, "trade_id")
        _required_text(item, "reason")
        if item.get("ownership") != "SYSTEM":
            raise AIRuntimeError("CANDIDATE_INVALID", "CLOSE is limited to explicitly SYSTEM-owned positions")
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
    multiplier = _positive_number(order.get("allocation_multiplier"), "allocation_multiplier")
    if multiplier not in {0.5, 0.75, 1.0}:
        raise AIRuntimeError(
            "ORDER_INVALID",
            "allocation_multiplier must be 0.5, 0.75, or 1.0",
        )
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


def _candidate_schema(kind: str) -> dict[str, Any]:
    common = [
        "schema_version", "run_id", "profile", "kind", "model",
        "reasoning_effort", "decided_at_utc", "source_digest", "thesis",
        "evidence_refs", "extensions",
    ]
    if kind == "trade":
        return {
            "required": common + ["action", "confidence", "orders", "position_actions", "requested_evidence"],
            "actions": sorted(TRADE_ACTIONS),
            "order_fields": [
                "decision_id", "pair", "side", "method", "vehicle", "order_type", "entry",
                "take_profit", "stop_loss", "units", "allocation_multiplier",
                "rationale", "extensions",
            ],
        }
    return {
        "required": common + ["regime", "risk_posture", "valid_until_utc", "themes", "instructions"],
        "risk_postures": sorted(REVIEW_POSTURES),
    }


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
