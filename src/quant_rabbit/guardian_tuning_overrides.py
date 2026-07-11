"""Evidence-bound runtime overrides for narrowly supported bot tuning."""

from __future__ import annotations

import copy
import fcntl
import functools
import hashlib
import inspect
import json
import math
import os
import re
import sqlite3
import threading
from contextlib import ContextDecorator
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.guardian_tuning_evaluator import MIN_FORWARD_SAMPLE_COUNT
from quant_rabbit.paths import ROOT


DEFAULT_OVERRIDE_PATH = ROOT / "data" / "guardian_tuning_overrides.json"
DEFAULT_WORK_ORDER_PATH = ROOT / "data" / "guardian_tuning_work_order.json"
MAX_OVERRIDE_BYTES = 256 * 1024
MAX_WORK_ORDER_BYTES = 4 * 1024 * 1024
MAX_ACTIVATION_MANIFEST_BYTES = 256 * 1024
MAX_TERMINAL_MANIFEST_BYTES = 512 * 1024
MAX_ACTIVATION_MANIFESTS = 1_000
MAX_ACTIVE_OVERRIDES = 100
MAX_PENDING_OVERRIDES = 100
MAX_OVERRIDE_HISTORY = 1_000
ACCEPTED_RESULT = "ACCEPTED_IMPROVEMENT"

class _RuntimeValidationCycle:
    """One revocable cache generation shared only by one validation phase."""

    def __init__(self) -> None:
        self.active = True
        self.generation = 0
        self.entries: dict[tuple[object, ...], dict[str, Any]] = {}
        self.lock = threading.RLock()

    def invalidate(self, *, close: bool = False) -> None:
        with self.lock:
            self.generation += 1
            self.entries = {}
            if close:
                self.active = False


_RUNTIME_VALIDATION_CACHE: ContextVar[_RuntimeValidationCycle | None] = ContextVar(
    "guardian_tuning_runtime_validation_cache",
    default=None,
)

_ACTIVATION_LEDGER_ANCHOR_FIELDS = frozenset(
    {
        "ledger_rowid_watermark",
        "ledger_prefix_sha256",
        "execution_ledger_coverage_start_utc",
        "last_oanda_transaction_id",
        "captured_at_utc",
    }
)
_ACTIVATION_RECORD_EXACT_FIELDS = (
    "work_order_id",
    "experiment_id",
    "experiment_contract_digest",
    "experiment_evidence_ref",
    "override_key",
    "pair",
    "method",
    "lane_id",
    "parameter",
    "previous_value",
    "candidate_value",
    "rollback_value",
    "terminal_confirmation_sha256",
    "activated_at_utc",
    "activation_ledger_anchor",
)


class guardian_tuning_validation_cycle(ContextDecorator):
    """Share one deep attestation only inside a single gateway validation phase."""

    def __init__(self) -> None:
        self._entries: list[tuple[Token[_RuntimeValidationCycle | None], _RuntimeValidationCycle]] = []

    def _recreate_cm(self) -> guardian_tuning_validation_cycle:
        return type(self)()

    def __enter__(self) -> guardian_tuning_validation_cycle:
        # Every decorated gateway invocation owns a distinct phase.  Inherited
        # ContextVars (async tasks/copy_context) and synchronous re-entry must
        # never join an older request's cache merely because one is visible.
        cycle = _RuntimeValidationCycle()
        token = _RUNTIME_VALIDATION_CACHE.set(cycle)
        self._entries.append((token, cycle))
        return self

    def __exit__(self, *exc_info: object) -> None:
        if not self._entries:
            return
        token, cycle = self._entries.pop()
        # Copied child contexts retain the cycle object after this context's
        # token reset.  Revocation therefore lives on the shared object too.
        cycle.invalidate(close=True)
        _RUNTIME_VALIDATION_CACHE.reset(token)

    def __call__(self, func: Any) -> Any:
        # ContextDecorator's stock wrapper exits as soon as an async function
        # returns its coroutine.  Keep the phase active across the actual await.
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_inner(*args: Any, **kwargs: Any) -> Any:
                with self._recreate_cm():
                    return await func(*args, **kwargs)

            return async_inner
        return super().__call__(func)


def clear_guardian_tuning_validation_cache() -> None:
    """Start a fresh attestation phase before the final pre-POST recheck."""

    cycle = _RUNTIME_VALIDATION_CACHE.get()
    if cycle is not None:
        cycle.invalidate()


def _lane_identity(lane_id: str) -> tuple[str, str]:
    raw = str(lane_id or "")
    parts = [part.strip() for part in raw.split(":")]
    if (
        len(parts) != 5
        or any(not part for part in parts)
        or parts[2].upper() not in {"LONG", "SHORT"}
        or parts[4].upper() not in {"MARKET", "LIMIT", "STOP"}
    ):
        raise ValueError("tuning lane id is invalid")
    canonical = ":".join(
        (
            parts[0].lower(),
            parts[1].upper(),
            parts[2].upper(),
            parts[3].upper(),
            parts[4].upper(),
        )
    )
    if raw != canonical:
        raise ValueError("tuning lane id is not canonical")
    return str(parts[1]).upper(), str(parts[3]).upper()


def _activation_ledger_anchor(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _ACTIVATION_LEDGER_ANCHOR_FIELDS:
        raise ValueError("activation ledger anchor schema is invalid")
    rowid = value.get("ledger_rowid_watermark")
    prefix_sha = str(value.get("ledger_prefix_sha256") or "")
    coverage = str(value.get("execution_ledger_coverage_start_utc") or "")
    last_id = str(value.get("last_oanda_transaction_id") or "")
    captured_at = str(value.get("captured_at_utc") or "")
    if (
        isinstance(rowid, bool)
        or not isinstance(rowid, int)
        or rowid <= 0
        or re.fullmatch(r"[0-9a-f]{64}", prefix_sha) is None
        or not last_id.isdigit()
    ):
        raise ValueError("activation ledger anchor identity is invalid")
    for label, timestamp in (
        ("coverage", coverage),
        ("capture", captured_at),
    ):
        normalized = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
        if "." in normalized:
            head, tail = normalized.split(".", 1)
            offset_at = next(
                (index for index, char in enumerate(tail) if char in "+-"),
                len(tail),
            )
            fraction, offset = tail[:offset_at], tail[offset_at:]
            normalized = f"{head}.{fraction[:6].ljust(6, '0')}{offset}"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            raise ValueError(f"activation ledger {label} timestamp is invalid") from None
        if parsed.tzinfo is None:
            raise ValueError(
                f"activation ledger {label} timestamp must be timezone-aware"
            )
    return {
        "ledger_rowid_watermark": rowid,
        "ledger_prefix_sha256": prefix_sha,
        "execution_ledger_coverage_start_utc": coverage,
        "last_oanda_transaction_id": last_id,
        "captured_at_utc": captured_at,
    }


def runtime_forecast_floor_binding(
    *,
    lane_id: str,
    environ: dict[str, str] | None = None,
    override_path: Path | None = None,
    queue_path: Path = DEFAULT_WORK_ORDER_PATH,
    allow_post_activation_monitor_pending: bool = False,
) -> dict[str, Any]:
    """Resolve the exact production setting controlled by a reviewed lane."""

    env = os.environ if environ is None else environ
    _, method = _lane_identity(lane_id)
    if method == "RANGE_ROTATION":
        variable = "QR_FORECAST_RANGE_ROTATION_MIN_CONFIDENCE"
        default = 0.50
    else:
        variable = "QR_FORECAST_DIRECTIONAL_LIVE_MIN_CONFIDENCE"
        default = 0.65
    try:
        value = float(env.get(variable, str(default)))
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"active setting {variable} is not numeric") from None
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"active setting {variable} is outside 0..1")
    base_value = value
    if override_path is not None:
        pair, _ = _lane_identity(lane_id)
        resolution = resolve_forecast_confidence_floor_state(
            pair=pair,
            method=method,
            lane_id=lane_id,
            fallback=base_value,
            path=override_path,
            queue_path=queue_path,
        )
        allowed_statuses = {"NO_OVERRIDE", "ACTIVE_OVERRIDE"}
        if allow_post_activation_monitor_pending:
            allowed_statuses.add("OVERRIDE_POST_ACTIVATION_MONITOR_PENDING")
        if resolution["status"] not in allowed_statuses:
            raise ValueError(
                f"guardian tuning override state is not ready: {resolution['status']}"
            )
        value = float(resolution["resolved_value"])
    return {
        "parameter": "forecast_confidence_floor",
        "environment_variable": variable,
        "resolved_value": value,
        "base_runtime_value": base_value,
        "binding_source": (
            "EVIDENCE_OVERRIDE" if value > base_value else "RUNTIME_ENVIRONMENT"
        ),
        "method": method,
    }


def _payload_digest(payload: dict[str, Any]) -> str:
    material = {key: value for key, value in payload.items() if key != "state_digest_sha256"}
    return hashlib.sha256(
        json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {
            "schema_version": 1,
            "active_overrides": [],
            "pending_overrides": [],
            "history": [],
        }
    if len(raw) > MAX_OVERRIDE_BYTES:
        raise ValueError("guardian tuning override state exceeds its bound")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("guardian tuning override state schema is invalid")
    active = payload.get("active_overrides")
    pending = payload.get("pending_overrides", [])
    history = payload.get("history")
    if (
        not isinstance(active, list)
        or len(active) > MAX_ACTIVE_OVERRIDES
        or not isinstance(pending, list)
        or len(pending) > MAX_PENDING_OVERRIDES
        or not isinstance(history, list)
        or len(history) > MAX_OVERRIDE_HISTORY
    ):
        raise ValueError("guardian tuning override collections are invalid")
    digest = str(payload.get("state_digest_sha256") or "")
    if digest != _payload_digest(payload):
        raise ValueError("guardian tuning override state digest is invalid")
    return {**payload, "pending_overrides": pending}


def _write(path: Path, payload: dict[str, Any]) -> None:
    material = {**payload, "state_digest_sha256": _payload_digest(payload)}
    raw = (
        json.dumps(material, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(raw) > MAX_OVERRIDE_BYTES:
        raise ValueError("guardian tuning override write exceeds its bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def read_stored_override_record(
    *,
    path: Path,
    override_key: str,
    experiment_id: str,
) -> dict[str, Any]:
    """Read one digest-valid record for lifecycle reconciliation, not permission."""

    state = _load(path)
    matches = [
        dict(item)
        for item in state.get("active_overrides", [])
        if isinstance(item, dict)
        and str(item.get("override_key") or "") == str(override_key or "")
        and str(item.get("experiment_id") or "") == str(experiment_id or "")
    ]
    if len(matches) != 1:
        raise ValueError("stored guardian tuning override is missing or ambiguous")
    return matches[0]


def read_active_override_records(
    *,
    path: Path = DEFAULT_OVERRIDE_PATH,
) -> list[dict[str, Any]]:
    """Return digest-valid active/quarantined records for the monitor worker."""

    state = _load(path)
    return [
        dict(item)
        for item in state.get("active_overrides", [])
        if isinstance(item, dict)
    ]


def write_terminal_commitment_manifest(
    *,
    queue_path: Path,
    terminal: dict[str, Any],
) -> str:
    """Persist the complete accepted terminal beyond bounded queue history."""

    raw = (
        json.dumps(terminal, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(raw) > MAX_TERMINAL_MANIFEST_BYTES:
        raise ValueError("guardian tuning terminal manifest exceeds its bound")
    digest = hashlib.sha256(raw).hexdigest()
    directory = queue_path.parent / "guardian_tuning_terminal_manifests"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{digest}.json"
    try:
        with destination.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if destination.read_bytes() != raw:
            raise ValueError("terminal manifest content address conflicts") from None
    return f"data/guardian_tuning_terminal_manifests/{digest}.json#sha256={digest}"


def _read_terminal_commitment_manifest(
    *,
    queue_path: Path,
    ref: object,
) -> dict[str, Any]:
    match = re.fullmatch(
        r"data/guardian_tuning_terminal_manifests/([0-9a-f]{64})\.json"
        r"#sha256=([0-9a-f]{64})",
        str(ref or ""),
    )
    if match is None or match.group(1) != match.group(2):
        raise ValueError("terminal manifest ref is invalid")
    digest = match.group(1)
    path = queue_path.parent / "guardian_tuning_terminal_manifests" / f"{digest}.json"
    raw = path.read_bytes()
    if len(raw) > MAX_TERMINAL_MANIFEST_BYTES or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("terminal manifest content address is invalid")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("terminal manifest schema is invalid")
    return payload


def _strict_queue_source_sha256(queue_path: Path) -> str:
    """Reuse the lifecycle writer's full queue and evidence-chain validator."""

    try:
        from tools import guardian_wake_dispatcher as dispatcher
    except (ImportError, RuntimeError) as exc:
        raise ValueError(f"strict tuning queue validator is unavailable: {exc}") from None
    loaded = dispatcher._load_tuning_work_order(queue_path)
    if loaded.get("_read_error"):
        raise ValueError(f"strict tuning queue validation failed: {loaded['_read_error']}")
    source_sha256 = str(loaded.get("_queue_source_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("strict tuning queue source digest is unavailable")
    return source_sha256


def _strict_terminal_evidence_error(
    *,
    queue_path: Path,
    terminal: dict[str, Any],
) -> str | None:
    """Revalidate immutable terminal evidence even after queue history trims it."""

    try:
        from tools import guardian_wake_dispatcher as dispatcher
    except (ImportError, RuntimeError) as exc:
        return f"strict tuning evidence validator is unavailable: {exc}"
    return dispatcher._tuning_queue_terminal_evidence_error(
        queue_path,
        {
            "queue_schema_revision": 4,
            "terminal_history": [terminal],
        },
    )


def _canonical_terminal_digest(terminal: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            terminal,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _same_number(left: object, right: object) -> bool:
    try:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    except (OverflowError, TypeError, ValueError):
        return False


def _terminal_commitment_error(
    *,
    staged: dict[str, Any],
    terminal: dict[str, Any],
    head: dict[str, Any],
) -> str | None:
    application = (
        terminal.get("tuning_override_application")
        if isinstance(terminal.get("tuning_override_application"), dict)
        else {}
    )
    review = (
        terminal.get("bot_tuning_review")
        if isinstance(terminal.get("bot_tuning_review"), dict)
        else {}
    )
    adjustments = review.get("proposed_adjustments")
    adjustment = (
        adjustments[0]
        if isinstance(adjustments, list)
        and len(adjustments) == 1
        and isinstance(adjustments[0], dict)
        else {}
    )
    prepared = (
        terminal.get("prepared_experiment_contract")
        if isinstance(terminal.get("prepared_experiment_contract"), dict)
        else {}
    )
    source_identity = (
        prepared.get("source_identity")
        if isinstance(prepared.get("source_identity"), dict)
        else {}
    )
    binding = (
        prepared.get("active_parameter_binding")
        if isinstance(prepared.get("active_parameter_binding"), dict)
        else {}
    )
    lane_id = str(staged.get("lane_id") or "")
    parameter = str(staged.get("parameter") or "")
    expected_key = f"{lane_id}|{parameter}"
    try:
        pair, method = _lane_identity(lane_id)
    except ValueError:
        return "staged override lane is invalid"
    terminal_digest = _canonical_terminal_digest(terminal)
    if (
        parameter != "forecast_confidence_floor"
        or str(staged.get("override_key") or "") != expected_key
        or str(staged.get("pair") or "").upper() != pair
        or str(staged.get("method") or "").upper() != method
        or not _same_number(staged.get("rollback_value"), staged.get("previous_value"))
        or not _same_number(adjustment.get("current_value"), staged.get("previous_value"))
        or not _same_number(adjustment.get("candidate_value"), staged.get("candidate_value"))
        or not _same_number(binding.get("resolved_value"), staged.get("previous_value"))
        or not (
            0.0
            <= float(staged.get("previous_value"))
            < float(staged.get("candidate_value"))
            <= 1.0
        )
        or str(adjustment.get("pair") or "").upper() != pair
        or str(adjustment.get("lane_id") or "") != lane_id
        or str(adjustment.get("parameter") or "") != parameter
        or str(source_identity.get("lane_id") or "") != lane_id
        or str(binding.get("parameter") or "") != parameter
        or str(binding.get("method") or "").upper() != method
        or str(prepared.get("experiment_contract_digest") or "")
        != str(staged.get("experiment_contract_digest") or "")
    ):
        return "accepted terminal parameter contract conflicts with staged override"
    if (
        str(terminal.get("work_order_id") or "")
        != str(staged.get("work_order_id") or "")
        or str(terminal.get("status") or "").upper() != "CONSUMED"
        or str(terminal.get("experiment_id") or "")
        != str(staged.get("experiment_id") or "")
        or str(terminal.get("experiment_result") or "") != ACCEPTED_RESULT
        or str(terminal.get("experiment_evidence_ref") or "")
        != str(staged.get("experiment_evidence_ref") or "")
        or str(terminal.get("experiment_contract_digest") or "")
        != str(staged.get("experiment_contract_digest") or "")
        or str(terminal.get("terminal_transition_source") or "")
        != "guardian_tuning_work_order_lifecycle"
        or terminal.get("live_permission_allowed") is not False
        or terminal.get("no_direct_oanda") is not True
        or str(application.get("status") or "")
        not in {"OVERRIDE_STAGED", "OVERRIDE_ALREADY_STAGED"}
        or str(application.get("override_key") or "") != expected_key
        or not _same_number(application.get("candidate_value"), staged.get("candidate_value"))
        or application.get("activated_at_utc") != staged.get("activated_at_utc")
        or application.get("activation_ledger_anchor")
        != staged.get("activation_ledger_anchor")
    ):
        return "accepted terminal record conflicts with staged override"
    if (
        str(head.get("status") or "")
        not in {
            "ACTIVE_COMMITTED",
            "MONITORED_KEEP_COMMITTED",
            "QUARANTINED_COMMITTED",
        }
        or str(head.get("override_key") or "") != expected_key
        or str(head.get("work_order_id") or "")
        != str(staged.get("work_order_id") or "")
        or str(head.get("experiment_id") or "")
        != str(staged.get("experiment_id") or "")
        or str(head.get("experiment_result") or "") != ACCEPTED_RESULT
        or str(head.get("experiment_evidence_ref") or "")
        != str(staged.get("experiment_evidence_ref") or "")
        or str(head.get("experiment_contract_digest") or "")
        != str(staged.get("experiment_contract_digest") or "")
        or str(head.get("terminal_confirmation_sha256") or "") != terminal_digest
        or str(head.get("pair") or "").upper() != pair
        or str(head.get("method") or "").upper() != method
        or str(head.get("lane_id") or "") != lane_id
        or str(head.get("parameter") or "") != parameter
        or not _same_number(head.get("candidate_value"), staged.get("candidate_value"))
        or head.get("activated_at_utc") != staged.get("activated_at_utc")
        or head.get("activation_ledger_anchor")
        != staged.get("activation_ledger_anchor")
        or head.get("live_permission_allowed") is not False
        or head.get("no_direct_oanda") is not True
    ):
        return "durable override lifecycle commitment conflicts"
    return None


def _terminal_confirmation(
    *,
    queue_path: Path,
    staged: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    try:
        raw = queue_path.read_bytes()
    except FileNotFoundError:
        raise ValueError("terminal tuning queue is missing") from None
    if len(raw) > MAX_WORK_ORDER_BYTES:
        raise ValueError("terminal tuning queue exceeds its bound")
    strict_sha256 = _strict_queue_source_sha256(queue_path)
    if hashlib.sha256(raw).hexdigest() != strict_sha256:
        raise ValueError("terminal tuning queue changed during strict validation")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("queue_schema_revision") != 4:
        raise ValueError("terminal tuning queue revision is invalid")
    heads = payload.get("override_lifecycle_heads")
    matching_heads = [
        head
        for head in heads
        if isinstance(head, dict)
        and str(head.get("override_key") or "")
        == str(staged.get("override_key") or "")
        and str(head.get("experiment_id") or "")
        == str(staged.get("experiment_id") or "")
    ] if isinstance(heads, list) else []
    if len(matching_heads) != 1:
        raise ValueError("durable override lifecycle commitment is missing")
    head = matching_heads[0]
    terminal = _read_terminal_commitment_manifest(
        queue_path=queue_path,
        ref=head.get("terminal_record_ref"),
    )
    evidence_error = _strict_terminal_evidence_error(
        queue_path=queue_path,
        terminal=terminal,
    )
    if evidence_error is not None:
        raise ValueError(f"immutable terminal evidence is invalid: {evidence_error}")
    digest = _canonical_terminal_digest(terminal)
    history = payload.get("terminal_history")
    if not isinstance(history, list):
        raise ValueError("terminal tuning history is unavailable")
    matches = [
        item
        for item in history
        if isinstance(item, dict)
        and str(item.get("work_order_id") or "")
        == str(staged.get("work_order_id") or "")
        and str(item.get("experiment_id") or "")
        == str(staged.get("experiment_id") or "")
    ]
    if len(matches) > 1 or (matches and matches[0] != terminal):
        raise ValueError("accepted terminal history conflicts with immutable commitment")
    error = _terminal_commitment_error(
        staged=staged,
        terminal=terminal,
        head=head,
    )
    if error is not None:
        raise ValueError(error)
    return terminal, digest, head


def _write_activation_manifest(
    *,
    override_path: Path,
    staged: dict[str, Any],
    terminal: dict[str, Any],
    terminal_digest: str,
    lifecycle_head: dict[str, Any],
    now: datetime,
) -> str:
    payload = {
        "schema_version": 1,
        "status": "ACTIVATED",
        "work_order_id": staged.get("work_order_id"),
        "experiment_id": staged.get("experiment_id"),
        "experiment_contract_digest": staged.get("experiment_contract_digest"),
        "experiment_evidence_ref": staged.get("experiment_evidence_ref"),
        "override_key": staged.get("override_key"),
        "pair": staged.get("pair"),
        "method": staged.get("method"),
        "lane_id": staged.get("lane_id"),
        "parameter": staged.get("parameter"),
        "previous_value": staged.get("previous_value"),
        "candidate_value": staged.get("candidate_value"),
        "rollback_value": staged.get("rollback_value"),
        "activation_ledger_anchor": _activation_ledger_anchor(
            staged.get("activation_ledger_anchor")
        ),
        "terminal_confirmation_sha256": terminal_digest,
        "terminal_record_ref": lifecycle_head.get("terminal_record_ref"),
        "terminal_record": terminal,
        "monitor_contract": {
            "mode": "FIXED_FIRST_20_COMPLETE_POST_ACTIVATION",
            "minimum_resolved_samples": MIN_FORWARD_SAMPLE_COUNT,
            "primary_metric": "net_jpy_per_1000_units_per_opportunity",
            "keep_if": ">0",
            "non_positive_action": "QUARANTINE",
            "previous_value_relaxation_requires_separate_proof": True,
        },
        "activated_at_utc": staged.get("activated_at_utc"),
        "live_permission_allowed": False,
        "no_direct_oanda": True,
    }
    raw = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(raw) > MAX_ACTIVATION_MANIFEST_BYTES:
        raise ValueError("guardian tuning activation manifest exceeds its bound")
    digest = hashlib.sha256(raw).hexdigest()
    directory = override_path.parent / "guardian_tuning_activation_manifests"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{digest}.json"
    try:
        with destination.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if destination.read_bytes() != raw:
            raise ValueError("activation manifest content address conflicts") from None
    return f"data/guardian_tuning_activation_manifests/{digest}.json#sha256={digest}"


def _read_activation_manifest(
    *,
    override_path: Path,
    ref: object,
) -> dict[str, Any]:
    match = re.fullmatch(
        r"data/guardian_tuning_activation_manifests/([0-9a-f]{64})\.json#sha256=([0-9a-f]{64})",
        str(ref or ""),
    )
    if match is None or match.group(1) != match.group(2):
        raise ValueError("activation manifest ref is invalid")
    digest = match.group(1)
    path = override_path.parent / "guardian_tuning_activation_manifests" / f"{digest}.json"
    raw = path.read_bytes()
    if len(raw) > MAX_ACTIVATION_MANIFEST_BYTES or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("activation manifest content address is invalid")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("activation manifest schema is invalid")
    return payload


def _manifest_scope_exists(
    *,
    override_path: Path,
    pair: str,
    method: str,
    lane_id: str,
) -> bool:
    directory = override_path.parent / "guardian_tuning_activation_manifests"
    if not directory.exists():
        return False
    paths = sorted(directory.glob("*.json"))
    if len(paths) > MAX_ACTIVATION_MANIFESTS:
        raise ValueError("activation manifest registry exceeds its bound")
    for manifest_path in paths:
        if manifest_path.is_symlink() or not re.fullmatch(r"[0-9a-f]{64}\.json", manifest_path.name):
            raise ValueError("activation manifest registry contains an invalid path")
        digest = manifest_path.stem
        payload = _read_activation_manifest(
            override_path=override_path,
            ref=(
                f"data/guardian_tuning_activation_manifests/{digest}.json"
                f"#sha256={digest}"
            ),
        )
        if (
            str(payload.get("pair") or "").upper() == pair
            and str(payload.get("method") or "").upper() == method
            and (not lane_id or str(payload.get("lane_id") or "") == lane_id)
        ):
            return True
    return False


def _queue_scope_has_commitment(
    *,
    queue_path: Path,
    pair: str,
    method: str,
    lane_id: str,
) -> bool:
    """Find an accepted durable head even when mutable override files vanished."""

    try:
        raw = queue_path.read_bytes()
    except FileNotFoundError:
        return False
    if len(raw) > MAX_WORK_ORDER_BYTES:
        raise ValueError("tuning queue exceeds its bound")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("tuning queue commitment is invalid")
    if payload.get("queue_schema_revision") != 4:
        # Legacy queues predate durable override heads and therefore cannot
        # prove an accepted setting.  A head smuggled into an older revision is
        # corruption, not a commitment.
        if "override_lifecycle_heads" not in payload:
            return False
        raise ValueError("tuning queue commitment revision is invalid")
    heads = payload.get("override_lifecycle_heads")
    if not isinstance(heads, list) or len(heads) > MAX_ACTIVE_OVERRIDES:
        raise ValueError("tuning queue lifecycle heads are invalid")
    matching: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for head in heads:
        if not isinstance(head, dict):
            raise ValueError("tuning queue lifecycle head is invalid")
        key = str(head.get("override_key") or "")
        if not key or key in seen_keys:
            raise ValueError("tuning queue lifecycle head identity is invalid")
        seen_keys.add(key)
        head_pair = str(head.get("pair") or "").upper()
        head_method = str(head.get("method") or "").upper()
        head_lane = str(head.get("lane_id") or "").strip()
        try:
            canonical_pair, canonical_method = _lane_identity(head_lane)
            candidate = float(head.get("candidate_value"))
            _activation_ledger_anchor(head.get("activation_ledger_anchor"))
            activated_at = str(head.get("activated_at_utc") or "")
            activated_parsed = datetime.fromisoformat(
                activated_at[:-1] + "+00:00"
                if activated_at.endswith("Z")
                else activated_at
            )
        except (OverflowError, TypeError, ValueError):
            raise ValueError("tuning queue lifecycle head scope is invalid") from None
        if (
            str(head.get("status") or "")
            not in {
                "ACTIVE_COMMITTED",
                "MONITORED_KEEP_COMMITTED",
                "QUARANTINED_COMMITTED",
            }
            or head_pair != canonical_pair
            or head_method != canonical_method
            or str(head.get("parameter") or "") != "forecast_confidence_floor"
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(head.get("terminal_confirmation_sha256") or ""),
            )
            or not re.fullmatch(
                r"data/guardian_tuning_terminal_manifests/[0-9a-f]{64}\.json"
                r"#sha256=[0-9a-f]{64}",
                str(head.get("terminal_record_ref") or ""),
            )
            or not math.isfinite(candidate)
            or not 0.0 <= candidate <= 1.0
            or activated_parsed.tzinfo is None
            or head.get("live_permission_allowed") is not False
            or head.get("no_direct_oanda") is not True
        ):
            raise ValueError("tuning queue lifecycle head commitment is invalid")
        if (
            head_pair == pair
            and head_method == method
            and (not lane_id or head_lane == lane_id)
        ):
            matching.append(head)
    if len(matching) > 1:
        raise ValueError("tuning queue lifecycle commitment is ambiguous")
    return bool(matching)


def _read_override_lifecycle_head(
    *,
    queue_path: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    raw = queue_path.read_bytes()
    if len(raw) > MAX_WORK_ORDER_BYTES:
        raise ValueError("tuning queue exceeds its bound")
    strict_sha256 = _strict_queue_source_sha256(queue_path)
    if hashlib.sha256(raw).hexdigest() != strict_sha256:
        raise ValueError("tuning queue changed during strict validation")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("queue_schema_revision") != 4:
        raise ValueError("tuning queue commitment revision is invalid")
    heads = payload.get("override_lifecycle_heads")
    matches = [
        head
        for head in heads
        if isinstance(head, dict)
        and str(head.get("override_key") or "")
        == str(record.get("override_key") or "")
    ] if isinstance(heads, list) else []
    if len(matches) != 1:
        raise ValueError("tuning queue lifecycle head is missing or ambiguous")
    head = matches[0]
    history = payload.get("terminal_history")
    terminal_matches = [
        item
        for item in history
        if isinstance(item, dict)
        and str(item.get("work_order_id") or "")
        == str(record.get("work_order_id") or "")
        and str(item.get("experiment_id") or "")
        == str(record.get("experiment_id") or "")
    ] if isinstance(history, list) else []
    if len(terminal_matches) > 1 or (
        terminal_matches
        and _canonical_terminal_digest(terminal_matches[0])
        != str(head.get("terminal_confirmation_sha256") or "")
    ):
        raise ValueError("tuning queue terminal history conflicts with lifecycle head")
    if (
        str(head.get("status") or "")
        not in {
            "ACTIVE_COMMITTED",
            "MONITORED_KEEP_COMMITTED",
            "QUARANTINED_COMMITTED",
        }
        or str(head.get("work_order_id") or "")
        != str(record.get("work_order_id") or "")
        or str(head.get("experiment_id") or "")
        != str(record.get("experiment_id") or "")
        or str(head.get("experiment_evidence_ref") or "")
        != str(record.get("experiment_evidence_ref") or "")
        or str(head.get("experiment_contract_digest") or "")
        != str(record.get("experiment_contract_digest") or "")
        or str(head.get("terminal_confirmation_sha256") or "")
        != str(record.get("terminal_confirmation_sha256") or "")
        or str(head.get("lane_id") or "") != str(record.get("lane_id") or "")
        or head.get("activated_at_utc") != record.get("activated_at_utc")
        or head.get("activation_ledger_anchor")
        != record.get("activation_ledger_anchor")
        or float(head.get("candidate_value"))
        != float(record.get("candidate_value"))
    ):
        raise ValueError("tuning queue lifecycle head conflicts with override state")
    return {
        **head,
        "_terminal_evidence_validated_by_queue": bool(terminal_matches),
    }


def _active_provenance_error(
    *,
    record: dict[str, Any],
    manifest: dict[str, Any],
    head: dict[str, Any],
    queue_path: Path,
) -> str | None:
    terminal = (
        manifest.get("terminal_record")
        if isinstance(manifest.get("terminal_record"), dict)
        else {}
    )
    committed_terminal = _read_terminal_commitment_manifest(
        queue_path=queue_path,
        ref=head.get("terminal_record_ref"),
    )
    evidence_error = (
        None
        if head.get("_terminal_evidence_validated_by_queue") is True
        else _strict_terminal_evidence_error(
            queue_path=queue_path,
            terminal=committed_terminal,
        )
    )
    if evidence_error is not None:
        return f"immutable terminal evidence is invalid: {evidence_error}"
    terminal_digest = _canonical_terminal_digest(terminal)
    if (
        manifest.get("schema_version") != 1
        or str(manifest.get("status") or "") != "ACTIVATED"
        or any(
            manifest.get(field) != record.get(field)
            for field in _ACTIVATION_RECORD_EXACT_FIELDS
        )
        or manifest.get("terminal_record_ref") != head.get("terminal_record_ref")
        or terminal != committed_terminal
        or terminal_digest != str(record.get("terminal_confirmation_sha256") or "")
        or manifest.get("live_permission_allowed") is not False
        or manifest.get("no_direct_oanda") is not True
    ):
        return "active override manifest conflicts with committed provenance"
    head_status = str(head.get("status") or "")
    if head_status == "ACTIVE_COMMITTED":
        if record.get("activation_status") != "ACTIVE" or record.get(
            "monitor_decision"
        ) not in {None, ""}:
            return "active override state conflicts with unmonitored lifecycle head"
    else:
        expected_decision = (
            "KEEP" if head_status == "MONITORED_KEEP_COMMITTED" else "QUARANTINE"
        )
        expected_activation_status = (
            "ACTIVE" if expected_decision == "KEEP" else "QUARANTINED"
        )
        if (
            record.get("activation_status") != expected_activation_status
            or record.get("monitor_decision") != expected_decision
            or record.get("monitor_evidence_ref")
            != head.get("monitor_evidence_ref")
            or not _same_number(
                record.get("post_activation_primary_metric"),
                head.get("post_activation_primary_metric"),
            )
            or record.get("monitored_at_utc") != head.get("monitored_at_utc")
        ):
            return "monitor decision state conflicts with durable lifecycle head"
        from quant_rabbit.guardian_tuning_monitor import (
            validate_post_activation_monitor_evidence,
        )

        validation = validate_post_activation_monitor_evidence(
            queue_path=queue_path,
            ledger_path=queue_path.with_name("execution_ledger.db"),
            evidence_ref=record.get("monitor_evidence_ref"),
            expected_record=record,
        )
        if (
            validation.get("status") != "VALID"
            or validation.get("decision") != expected_decision
            or not _same_number(
                validation.get("primary_metric_value"),
                record.get("post_activation_primary_metric"),
            )
        ):
            return "post-activation monitor evidence is invalid"
    return _terminal_commitment_error(
        staged=record,
        terminal=terminal,
        head=head,
    )


def read_validated_kept_predecessor_record(
    *,
    path: Path,
    queue_path: Path,
    override_key: str,
    experiment_id: str,
) -> dict[str, Any]:
    """Validate the exact prior KEEP while ignoring a staged successor.

    The ordinary runtime resolver deliberately reports a pending successor
    before its active predecessor.  Lifecycle retry needs the inverse view: it
    must prove the still-active predecessor and its still-current ledger truth
    without treating its own crash-left pending record as provenance failure.
    """

    record = read_stored_override_record(
        path=path,
        override_key=override_key,
        experiment_id=experiment_id,
    )
    if (
        record.get("activation_status") != "ACTIVE"
        or record.get("monitor_decision") != "KEEP"
    ):
        raise ValueError("predecessor is not an active monitored KEEP")
    manifest = _read_activation_manifest(
        override_path=path,
        ref=record.get("activation_manifest_ref"),
    )
    head = _read_override_lifecycle_head(
        queue_path=queue_path,
        record=record,
    )
    if str(head.get("status") or "") != "MONITORED_KEEP_COMMITTED":
        raise ValueError("predecessor queue head is not a monitored KEEP")
    provenance_error = _active_provenance_error(
        record=record,
        manifest=manifest,
        head=head,
        queue_path=queue_path,
    )
    if provenance_error is not None:
        raise ValueError(provenance_error)
    return record


def _kept_predecessor_provenance_error(
    *,
    record: dict[str, Any],
    override_path: Path,
    queue_path: Path,
) -> str | None:
    """Revalidate a superseded KEEP without relying on its replaced queue head."""

    if (
        record.get("activation_status") != "ACTIVE"
        or record.get("monitor_decision") != "KEEP"
    ):
        return "predecessor is not a monitored KEEP"
    try:
        manifest = _read_activation_manifest(
            override_path=override_path,
            ref=record.get("activation_manifest_ref"),
        )
        terminal = (
            manifest.get("terminal_record")
            if isinstance(manifest.get("terminal_record"), dict)
            else {}
        )
        if (
            manifest.get("schema_version") != 1
            or manifest.get("status") != "ACTIVATED"
            or any(
                manifest.get(field) != record.get(field)
                for field in _ACTIVATION_RECORD_EXACT_FIELDS
            )
            or _canonical_terminal_digest(terminal)
            != str(record.get("terminal_confirmation_sha256") or "")
            or manifest.get("live_permission_allowed") is not False
            or manifest.get("no_direct_oanda") is not True
        ):
            return "predecessor activation manifest conflicts"
        terminal_error = _strict_terminal_evidence_error(
            queue_path=queue_path,
            terminal=terminal,
        )
        if terminal_error is not None:
            return f"predecessor terminal evidence is invalid: {terminal_error}"
        from quant_rabbit.guardian_tuning_monitor import (
            validate_post_activation_monitor_evidence,
        )

        validation = validate_post_activation_monitor_evidence(
            queue_path=queue_path,
            ledger_path=queue_path.with_name("execution_ledger.db"),
            evidence_ref=record.get("monitor_evidence_ref"),
            expected_record=record,
        )
        if (
            validation.get("status") != "VALID"
            or validation.get("decision") != "KEEP"
            or not _same_number(
                validation.get("primary_metric_value"),
                record.get("post_activation_primary_metric"),
            )
        ):
            return "predecessor post-activation monitor evidence is invalid"
    except (
        OSError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        sqlite3.Error,
    ) as exc:
        return f"predecessor provenance is unreadable: {type(exc).__name__}: {exc}"
    return None


def apply_accepted_override(
    *,
    path: Path,
    work_order: dict[str, Any],
    prepared_contract: dict[str, Any],
    experiment_id: str,
    experiment_result: str,
    evidence_ref: str,
    activation_ledger_anchor: dict[str, Any],
    now: datetime,
    queue_path: Path | None = None,
) -> dict[str, Any]:
    """Serialize override-state writers even outside the queue lifecycle tool."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("guardian tuning override state has a concurrent writer") from None
        return _apply_accepted_override_locked(
            path=path,
            work_order=work_order,
            prepared_contract=prepared_contract,
            experiment_id=experiment_id,
            experiment_result=experiment_result,
            evidence_ref=evidence_ref,
            activation_ledger_anchor=activation_ledger_anchor,
            now=now,
            queue_path=queue_path,
        )


def _apply_accepted_override_locked(
    *,
    path: Path,
    work_order: dict[str, Any],
    prepared_contract: dict[str, Any],
    experiment_id: str,
    experiment_result: str,
    evidence_ref: str,
    activation_ledger_anchor: dict[str, Any],
    now: datetime,
    queue_path: Path | None,
) -> dict[str, Any]:
    """Stage a proven tightening; it remains dormant until queue commit."""

    if experiment_result != ACCEPTED_RESULT:
        return {"status": "NO_OVERRIDE_FOR_REJECTED_RESULT", "applied": False}
    ledger_anchor = _activation_ledger_anchor(activation_ledger_anchor)
    review = work_order.get("bot_tuning_review")
    adjustments = review.get("proposed_adjustments") if isinstance(review, dict) else None
    if not isinstance(adjustments, list) or len(adjustments) != 1:
        raise ValueError("accepted override requires one reviewed adjustment")
    adjustment = adjustments[0]
    if str(adjustment.get("parameter") or "") != "forecast_confidence_floor":
        raise ValueError("accepted override parameter is unsupported")
    source_identity = prepared_contract.get("source_identity")
    if not isinstance(source_identity, dict):
        raise ValueError("prepared contract does not bind canonical source identity")
    lane_id = str(source_identity.get("lane_id") or "")
    pair, method = _lane_identity(lane_id)
    if pair != str(adjustment.get("pair") or "").upper():
        raise ValueError("accepted override lane/pair mismatch")
    if lane_id != str(adjustment.get("lane_id") or "").strip():
        raise ValueError("accepted override lane was not fixed by the review")
    current = float(adjustment.get("current_value"))
    candidate = float(adjustment.get("candidate_value"))
    if not (0.0 <= current < candidate <= 1.0):
        raise ValueError("accepted override must tighten a confidence floor")
    state = _load(path)
    active = [dict(item) for item in state.get("active_overrides", []) if isinstance(item, dict)]
    pending = [
        dict(item)
        for item in state.get("pending_overrides", [])
        if isinstance(item, dict)
    ]
    history = [dict(item) for item in state.get("history", []) if isinstance(item, dict)]
    key = f"{lane_id}|forecast_confidence_floor"
    already_active = next(
        (
            item
            for item in active
            if item.get("override_key") == key
            and str(item.get("experiment_id") or "") == experiment_id
        ),
        None,
    )
    if already_active is not None:
        return {
            "status": "OVERRIDE_ALREADY_ACTIVE",
            "applied": True,
            "override_key": key,
            "candidate_value": already_active.get("candidate_value"),
            "activated_at_utc": already_active.get("activated_at_utc"),
            "activation_ledger_anchor": already_active.get(
                "activation_ledger_anchor"
            ),
        }
    already_staged = next(
        (
            item
            for item in pending
            if item.get("override_key") == key
            and str(item.get("experiment_id") or "") == experiment_id
        ),
        None,
    )
    if already_staged is not None:
        return {
            "status": "OVERRIDE_ALREADY_STAGED",
            "applied": False,
            "staged": True,
            "override_key": key,
            "candidate_value": already_staged.get("candidate_value"),
            "activated_at_utc": already_staged.get("activated_at_utc"),
            "activation_ledger_anchor": already_staged.get(
                "activation_ledger_anchor"
            ),
        }
    if any(item.get("override_key") == key for item in pending):
        raise ValueError("another override activation is already pending for this lane")
    existing = next((item for item in active if item.get("override_key") == key), None)
    if existing is not None and (
        str(existing.get("activation_status") or "") != "ACTIVE"
        or str(existing.get("monitor_decision") or "") != "KEEP"
    ):
        raise ValueError(
            "prior override post-activation monitor was not kept; "
            "a successor tightening is forbidden"
        )
    if existing is not None:
        if queue_path is None:
            raise ValueError(
                "successor tightening requires the prior durable queue provenance"
            )
        prior_resolution = resolve_forecast_confidence_floor_state(
            pair=pair,
            method=method,
            lane_id=lane_id,
            fallback=0.0,
            path=path,
            queue_path=queue_path,
        )
        prior_record = prior_resolution.get("override")
        if (
            prior_resolution.get("status") != "ACTIVE_OVERRIDE"
            or not isinstance(prior_record, dict)
            or str(prior_record.get("monitor_decision") or "") != "KEEP"
            or str(prior_record.get("experiment_id") or "")
            != str(existing.get("experiment_id") or "")
        ):
            raise ValueError(
                "prior override current provenance is invalid; successor is forbidden"
            )
    expected_current = (
        float(existing.get("candidate_value"))
        if existing is not None
        else float(runtime_forecast_floor_binding(lane_id=lane_id)["resolved_value"])
    )
    prepared_current = float(
        prepared_contract["active_parameter_binding"]["resolved_value"]
    )
    if not math.isclose(
        prepared_current,
        expected_current,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("prepared current binding no longer matches runtime")
    if not math.isclose(current, expected_current, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("accepted override current value is no longer active")
    record = {
        "override_key": key,
        "activation_status": "PENDING_QUEUE_COMMIT",
        "work_order_id": str(work_order.get("work_order_id") or "").strip(),
        "pair": pair,
        "method": method,
        "lane_id": lane_id,
        "bot_family": str(adjustment.get("bot_family") or "").lower(),
        "parameter": "forecast_confidence_floor",
        "previous_value": current,
        "candidate_value": candidate,
        "rollback_value": current,
        "rollback_contract": {
            "minimum_forward_samples": MIN_FORWARD_SAMPLE_COUNT,
            "quarantine_if_post_activation_primary_metric_non_positive": True,
            "automatic_relaxation_to_previous_value": False,
        },
        "experiment_id": experiment_id,
        "experiment_contract_digest": prepared_contract.get(
            "experiment_contract_digest"
        ),
        "experiment_evidence_ref": evidence_ref,
        "staged_at_utc": now.astimezone(timezone.utc).isoformat(),
        "activated_at_utc": ledger_anchor["captured_at_utc"],
        "activation_ledger_anchor": ledger_anchor,
        "live_permission_allowed": False,
        "no_direct_oanda": True,
    }
    if not record["work_order_id"]:
        raise ValueError("accepted override work_order_id is required")
    pending.append(record)
    if len(pending) > MAX_PENDING_OVERRIDES:
        raise ValueError("guardian tuning pending override capacity is full")
    _write(
        path,
        {
            "schema_version": 1,
            "updated_at_utc": now.astimezone(timezone.utc).isoformat(),
            "active_overrides": active,
            "pending_overrides": pending,
            "history": history[:MAX_OVERRIDE_HISTORY],
        },
    )
    return {
        "status": "OVERRIDE_STAGED",
        "applied": False,
        "staged": True,
        "override_key": key,
        "previous_value": current,
        "candidate_value": candidate,
        "rollback_value": current,
        "activated_at_utc": record["activated_at_utc"],
        "activation_ledger_anchor": ledger_anchor,
    }


def confirm_accepted_override(
    *,
    path: Path,
    queue_path: Path,
    work_order_id: str,
    experiment_id: str,
    experiment_result: str,
    evidence_ref: str,
    now: datetime,
) -> dict[str, Any]:
    """Serialize confirmation writes against staging and other confirmations."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("guardian tuning override state has a concurrent writer") from None
        return _confirm_accepted_override_locked(
            path=path,
            queue_path=queue_path,
            work_order_id=work_order_id,
            experiment_id=experiment_id,
            experiment_result=experiment_result,
            evidence_ref=evidence_ref,
            now=now,
        )


def _confirm_accepted_override_locked(
    *,
    path: Path,
    queue_path: Path,
    work_order_id: str,
    experiment_id: str,
    experiment_result: str,
    evidence_ref: str,
    now: datetime,
) -> dict[str, Any]:
    """Activate a staged tightening only after its terminal queue write."""

    if experiment_result != ACCEPTED_RESULT:
        return {"status": "NO_OVERRIDE_FOR_REJECTED_RESULT", "applied": False}
    state = _load(path)
    active = [dict(item) for item in state.get("active_overrides", []) if isinstance(item, dict)]
    pending = [
        dict(item)
        for item in state.get("pending_overrides", [])
        if isinstance(item, dict)
    ]
    history = [dict(item) for item in state.get("history", []) if isinstance(item, dict)]
    confirmed = next(
        (item for item in active if str(item.get("experiment_id") or "") == experiment_id),
        None,
    )
    if confirmed is not None:
        if (
            str(confirmed.get("work_order_id") or "") != work_order_id
            or str(confirmed.get("experiment_evidence_ref") or "") != evidence_ref
        ):
            raise ValueError("active override confirmation identity conflicts")
        terminal, terminal_digest, lifecycle_head = _terminal_confirmation(
            queue_path=queue_path,
            staged=confirmed,
        )
        manifest_ref = str(confirmed.get("activation_manifest_ref") or "")
        if manifest_ref:
            manifest = _read_activation_manifest(
                override_path=path,
                ref=manifest_ref,
            )
            provenance_error = _active_provenance_error(
                record=confirmed,
                manifest=manifest,
                head=lifecycle_head,
                queue_path=queue_path,
            )
            if provenance_error is not None:
                raise ValueError(provenance_error)
        else:
            manifest_ref = _write_activation_manifest(
                override_path=path,
                staged=confirmed,
                terminal=terminal,
                terminal_digest=terminal_digest,
                lifecycle_head=lifecycle_head,
                now=now,
            )
        if (
            confirmed.get("confirmed_after_terminal_write") is not True
            or confirmed.get("terminal_confirmation_sha256") != terminal_digest
            or confirmed.get("activation_manifest_ref") != manifest_ref
        ):
            active.remove(confirmed)
            confirmed = {
                **confirmed,
                "activation_status": "ACTIVE",
                "confirmed_after_terminal_write": True,
                "terminal_confirmation_sha256": terminal_digest,
                "activation_manifest_ref": manifest_ref,
                "activated_at_utc": confirmed.get("activated_at_utc"),
            }
            active.append(confirmed)
            _write(
                path,
                {
                    "schema_version": 1,
                    "updated_at_utc": now.astimezone(timezone.utc).isoformat(),
                    "active_overrides": active,
                    "pending_overrides": pending,
                    "history": history[:MAX_OVERRIDE_HISTORY],
                },
            )
        return {
            "status": "OVERRIDE_ALREADY_ACTIVE",
            "applied": True,
            "override_key": confirmed.get("override_key"),
            "candidate_value": confirmed.get("candidate_value"),
        }
    staged_matches = [
        item
        for item in pending
        if str(item.get("experiment_id") or "") == experiment_id
    ]
    if len(staged_matches) != 1:
        raise ValueError("accepted override stage is missing or ambiguous")
    staged = staged_matches[0]
    if (
        str(staged.get("work_order_id") or "") != work_order_id
        or str(staged.get("experiment_evidence_ref") or "") != evidence_ref
    ):
        raise ValueError("staged override confirmation identity conflicts")
    terminal, terminal_digest, lifecycle_head = _terminal_confirmation(
        queue_path=queue_path,
        staged=staged,
    )
    manifest_ref = _write_activation_manifest(
        override_path=path,
        staged=staged,
        terminal=terminal,
        terminal_digest=terminal_digest,
        lifecycle_head=lifecycle_head,
        now=now,
    )
    key = str(staged.get("override_key") or "")
    existing = next((item for item in active if item.get("override_key") == key), None)
    if existing is not None and (
        str(existing.get("activation_status") or "") != "ACTIVE"
        or str(existing.get("monitor_decision") or "") != "KEEP"
    ):
        raise ValueError(
            "prior override post-activation monitor was not kept; "
            "successor activation is forbidden"
        )
    if existing is not None:
        predecessor_error = _kept_predecessor_provenance_error(
            record=existing,
            override_path=path,
            queue_path=queue_path,
        )
        if predecessor_error is not None:
            raise ValueError(
                "prior override current provenance is invalid; "
                f"successor activation is forbidden: {predecessor_error}"
            )
    expected_current = (
        float(existing.get("candidate_value"))
        if existing is not None
        else float(
            runtime_forecast_floor_binding(
                lane_id=str(staged.get("lane_id") or "")
            )["resolved_value"]
        )
    )
    if not math.isclose(
        float(staged.get("previous_value")),
        expected_current,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("staged override current value is no longer active")
    if existing is not None:
        active.remove(existing)
        history.insert(
            0,
            {
                **existing,
                "activation_status": "SUPERSEDED",
                "superseded_at_utc": now.astimezone(timezone.utc).isoformat(),
                "superseded_by_experiment_id": experiment_id,
            },
        )
    pending.remove(staged)
    activated = {
        **staged,
        "activation_status": "ACTIVE",
        "confirmed_after_terminal_write": True,
        "terminal_confirmation_sha256": terminal_digest,
        "activation_manifest_ref": manifest_ref,
        "activated_at_utc": staged.get("activated_at_utc"),
    }
    active.append(activated)
    if len(active) > MAX_ACTIVE_OVERRIDES:
        raise ValueError("guardian tuning active override capacity is full")
    _write(
        path,
        {
            "schema_version": 1,
            "updated_at_utc": now.astimezone(timezone.utc).isoformat(),
            "active_overrides": active,
            "pending_overrides": pending,
            "history": history[:MAX_OVERRIDE_HISTORY],
        },
    )
    return {
        "status": "OVERRIDE_ACTIVATED",
        "applied": True,
        "override_key": key,
        "previous_value": staged.get("previous_value"),
        "candidate_value": staged.get("candidate_value"),
        "rollback_value": staged.get("rollback_value"),
        "activated_at_utc": staged.get("activated_at_utc"),
        "activation_ledger_anchor": staged.get("activation_ledger_anchor"),
    }


def confirm_post_activation_monitor(
    *,
    path: Path,
    queue_path: Path,
    override_key: str,
    experiment_id: str,
    monitor_evidence_ref: str,
    decision: str,
    primary_metric_value: float,
    now: datetime,
) -> dict[str, Any]:
    """Confirm a durable first-20 KEEP or fail-closed QUARANTINE decision."""

    normalized_decision = str(decision or "").upper()
    metric = float(primary_metric_value)
    if (
        normalized_decision not in {"KEEP", "QUARANTINE"}
        or not math.isfinite(metric)
        or normalized_decision != ("KEEP" if metric > 0.0 else "QUARANTINE")
    ):
        raise ValueError("post-activation monitor decision conflicts with its metric")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("guardian tuning override state has a concurrent writer") from None
        state = _load(path)
        active = [
            dict(item)
            for item in state.get("active_overrides", [])
            if isinstance(item, dict)
        ]
        pending = [
            dict(item)
            for item in state.get("pending_overrides", [])
            if isinstance(item, dict)
        ]
        history = [
            dict(item) for item in state.get("history", []) if isinstance(item, dict)
        ]
        matches = [
            item
            for item in active
            if str(item.get("override_key") or "") == str(override_key or "")
            and str(item.get("experiment_id") or "") == str(experiment_id or "")
        ]
        if len(matches) != 1:
            raise ValueError("post-activation monitor active override is missing or ambiguous")
        record = matches[0]
        head = _read_override_lifecycle_head(
            queue_path=queue_path,
            record=record,
        )
        expected_head_status = (
            "MONITORED_KEEP_COMMITTED"
            if normalized_decision == "KEEP"
            else "QUARANTINED_COMMITTED"
        )
        if (
            str(head.get("status") or "") != expected_head_status
            or head.get("monitor_decision") != normalized_decision
            or head.get("monitor_evidence_ref") != monitor_evidence_ref
            or not _same_number(
                head.get("post_activation_primary_metric"),
                metric,
            )
        ):
            raise ValueError("post-activation monitor queue commitment conflicts")
        monitored_at = str(head.get("monitored_at_utc") or "")
        updated_record = {
            **record,
            "activation_status": (
                "ACTIVE" if normalized_decision == "KEEP" else "QUARANTINED"
            ),
            "monitor_decision": normalized_decision,
            "monitor_evidence_ref": monitor_evidence_ref,
            "post_activation_primary_metric": metric,
            "monitored_at_utc": monitored_at,
        }
        manifest = _read_activation_manifest(
            override_path=path,
            ref=record.get("activation_manifest_ref"),
        )
        provenance_error = _active_provenance_error(
            record=updated_record,
            manifest=manifest,
            head=head,
            queue_path=queue_path,
        )
        if provenance_error is not None:
            raise ValueError(provenance_error)
        if record == updated_record:
            return {
                "status": "POST_ACTIVATION_MONITOR_ALREADY_CONFIRMED",
                "decision": normalized_decision,
                "override_key": override_key,
            }
        active.remove(record)
        active.append(updated_record)
        _write(
            path,
            {
                "schema_version": 1,
                "updated_at_utc": now.astimezone(timezone.utc).isoformat(),
                "active_overrides": active,
                "pending_overrides": pending,
                "history": history[:MAX_OVERRIDE_HISTORY],
            },
        )
        return {
            "status": (
                "POST_ACTIVATION_OVERRIDE_KEPT"
                if normalized_decision == "KEEP"
                else "POST_ACTIVATION_LANE_QUARANTINED"
            ),
            "decision": normalized_decision,
            "override_key": override_key,
            "primary_metric_value": metric,
        }


def reconcile_pending_overrides(
    *,
    path: Path = DEFAULT_OVERRIDE_PATH,
    queue_path: Path = DEFAULT_WORK_ORDER_PATH,
    now: datetime,
) -> dict[str, Any]:
    """Recover stage/terminal split states without granting trade permission."""

    queue_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = queue_path.with_name(f"{queue_path.name}.lock")
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "status": "OVERRIDE_RECONCILIATION_CONCURRENT_UPDATE",
                "reconciled_count": 0,
                "live_permission_allowed": False,
                "retry_required": True,
            }
        return _reconcile_pending_overrides_locked(
            path=path,
            queue_path=queue_path,
            now=now,
        )


def _reconcile_pending_overrides_locked(
    *,
    path: Path,
    queue_path: Path,
    now: datetime,
) -> dict[str, Any]:
    state = _load(path)
    pending = [
        dict(item)
        for item in state.get("pending_overrides", [])
        if isinstance(item, dict)
    ]
    if not pending:
        return {
            "status": "NO_PENDING_OVERRIDE_CONFIRMATIONS",
            "reconciled_count": 0,
            "pending_count": 0,
            "live_permission_allowed": False,
        }
    results: list[dict[str, Any]] = []
    for staged in pending:
        try:
            result = confirm_accepted_override(
                path=path,
                queue_path=queue_path,
                work_order_id=str(staged.get("work_order_id") or ""),
                experiment_id=str(staged.get("experiment_id") or ""),
                experiment_result=ACCEPTED_RESULT,
                evidence_ref=str(staged.get("experiment_evidence_ref") or ""),
                now=now,
            )
            results.append(result)
        except (
            OSError,
            OverflowError,
            RecursionError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            sqlite3.Error,
        ) as exc:
            results.append(
                {
                    "status": "OVERRIDE_CONFIRMATION_STILL_PENDING",
                    "experiment_id": staged.get("experiment_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    remaining = _load(path).get("pending_overrides", [])
    return {
        "status": (
            "PENDING_OVERRIDES_RECONCILED"
            if not remaining
            else "OVERRIDE_CONFIRMATION_PENDING"
        ),
        "reconciled_count": sum(
            1
            for item in results
            if item.get("status") in {"OVERRIDE_ACTIVATED", "OVERRIDE_ALREADY_ACTIVE"}
        ),
        "pending_count": len(remaining),
        "results": results,
        "live_permission_allowed": False,
    }


def _resolve_forecast_confidence_floor_state_uncached(
    *,
    pair: str,
    method: str,
    lane_id: str | None = None,
    fallback: float,
    path: Path = DEFAULT_OVERRIDE_PATH,
    queue_path: Path = DEFAULT_WORK_ORDER_PATH,
) -> dict[str, Any]:
    """Resolve an override while distinguishing absence from corrupt state."""

    state_missing = not path.exists()
    try:
        state = _load(path)
    except FileNotFoundError:
        return {"status": "NO_OVERRIDE", "resolved_value": fallback}
    except (OSError, RecursionError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "OVERRIDE_STATE_INVALID",
            "resolved_value": fallback,
            "error": f"{type(exc).__name__}: {exc}",
        }
    normalized_pair = str(pair or "").upper()
    normalized_method = str(method or "").upper()
    exact_lane_id = str(lane_id or "").strip()
    def external_commitment(scope_lane_id: str) -> bool:
        has_manifest = _manifest_scope_exists(
            override_path=path,
            pair=normalized_pair,
            method=normalized_method,
            lane_id=scope_lane_id,
        )
        has_queue_commitment = _queue_scope_has_commitment(
            queue_path=queue_path,
            pair=normalized_pair,
            method=normalized_method,
            lane_id=scope_lane_id,
        )
        return has_manifest or has_queue_commitment

    if state_missing:
        try:
            has_external_commitment = external_commitment(exact_lane_id)
        except (OSError, RecursionError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return {
                "status": "OVERRIDE_STATE_INVALID",
                "resolved_value": fallback,
                "error": f"{type(exc).__name__}: {exc}",
            }
        if has_external_commitment:
            return {
                "status": "OVERRIDE_STATE_MISSING_WITH_COMMITMENT",
                "resolved_value": fallback,
            }
    relevant = [
        item
        for collection in (
            state.get("active_overrides", []),
            state.get("pending_overrides", []),
        )
        for item in collection
        if isinstance(item, dict)
        and str(item.get("pair") or "").upper() == normalized_pair
        and str(item.get("method") or "").upper() == normalized_method
    ]
    if not exact_lane_id:
        try:
            has_external_scope = external_commitment("")
        except (OSError, RecursionError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return {
                "status": "OVERRIDE_STATE_INVALID",
                "resolved_value": fallback,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "status": (
                "OVERRIDE_LANE_ID_REQUIRED"
                if relevant or has_external_scope
                else "NO_OVERRIDE"
            ),
            "resolved_value": fallback,
        }
    try:
        lane_pair, lane_method = _lane_identity(exact_lane_id)
    except ValueError:
        try:
            has_external_scope = external_commitment("")
        except (OSError, RecursionError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return {
                "status": "OVERRIDE_STATE_INVALID",
                "resolved_value": fallback,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "status": (
                "OVERRIDE_LANE_ID_INVALID"
                if relevant or has_external_scope
                else "NO_OVERRIDE"
            ),
            "resolved_value": fallback,
        }
    if lane_pair != normalized_pair or lane_method != normalized_method:
        return {"status": "OVERRIDE_LANE_ID_INVALID", "resolved_value": fallback}
    key = f"{exact_lane_id}|forecast_confidence_floor"
    pending_candidates = [
        item
        for item in state.get("pending_overrides", [])
        if isinstance(item, dict) and item.get("override_key") == key
    ]
    if pending_candidates:
        return {
            "status": "OVERRIDE_CONFIRMATION_PENDING",
            "resolved_value": fallback,
        }
    candidates = [
        item
        for item in state.get("active_overrides", [])
        if isinstance(item, dict) and item.get("override_key") == key
    ]
    if not candidates:
        try:
            has_external_commitment = external_commitment(exact_lane_id)
        except (OSError, RecursionError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return {
                "status": "OVERRIDE_STATE_INVALID",
                "resolved_value": fallback,
                "error": f"{type(exc).__name__}: {exc}",
            }
        if has_external_commitment:
            return {
                "status": "OVERRIDE_STATE_MISSING_WITH_COMMITMENT",
                "resolved_value": fallback,
            }
        return {"status": "NO_OVERRIDE", "resolved_value": fallback}
    if len(candidates) != 1:
        return {"status": "OVERRIDE_STATE_INVALID", "resolved_value": fallback}
    record = candidates[0]
    try:
        candidate = float(record["candidate_value"])
    except (OverflowError, KeyError, TypeError, ValueError):
        return {"status": "OVERRIDE_STATE_INVALID", "resolved_value": fallback}
    if (
        record.get("activation_status") not in {"ACTIVE", "QUARANTINED"}
        or record.get("confirmed_after_terminal_write") is not True
        or record.get("live_permission_allowed") is not False
        or record.get("no_direct_oanda") is not True
        or not str(record.get("work_order_id") or "")
        or len(str(record.get("terminal_confirmation_sha256") or "")) != 64
        or not math.isfinite(candidate)
        or not 0.0 <= candidate <= 1.0
    ):
        return {"status": "OVERRIDE_STATE_INVALID", "resolved_value": fallback}
    try:
        manifest = _read_activation_manifest(
            override_path=path,
            ref=record.get("activation_manifest_ref"),
        )
        head = _read_override_lifecycle_head(
            queue_path=queue_path,
            record=record,
        )
        provenance_error = _active_provenance_error(
            record=record,
            manifest=manifest,
            head=head,
            queue_path=queue_path,
        )
        if provenance_error is not None:
            raise ValueError(provenance_error)
    except (OSError, RecursionError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "OVERRIDE_STATE_INVALID",
            "resolved_value": fallback,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if record.get("activation_status") == "QUARANTINED":
        return {
            "status": "OVERRIDE_LANE_QUARANTINED",
            "resolved_value": max(float(fallback), candidate),
            "override": record,
        }
    if not str(record.get("monitor_decision") or ""):
        try:
            from quant_rabbit.guardian_tuning_cohort import (
                build_post_activation_monitor_cohort,
            )

            monitor_cohort = build_post_activation_monitor_cohort(
                ledger_path=queue_path.with_name("execution_ledger.db"),
                lane_id=str(record.get("lane_id") or ""),
                activated_at_utc=record.get("activated_at_utc"),
                activation_ledger_anchor=record.get("activation_ledger_anchor"),
            )
        except (
            OSError,
            OverflowError,
            RecursionError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            sqlite3.Error,
        ) as exc:
            return {
                "status": "OVERRIDE_STATE_INVALID",
                "resolved_value": fallback,
                "error": (
                    "post-activation monitor gate is unreadable: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        cohort_status = str(monitor_cohort.get("status") or "")
        if cohort_status == "WAITING_FOR_FIRST_20_ENTRIES":
            entry_count = monitor_cohort.get("entry_count")
            required_count = monitor_cohort.get("required_entry_count")
            if (
                isinstance(entry_count, bool)
                or not isinstance(entry_count, int)
                or not 0 <= entry_count < MIN_FORWARD_SAMPLE_COUNT
                or required_count != MIN_FORWARD_SAMPLE_COUNT
            ):
                return {
                    "status": "OVERRIDE_STATE_INVALID",
                    "resolved_value": fallback,
                    "error": "post-activation monitor entry count is invalid",
                }
        if cohort_status != "WAITING_FOR_FIRST_20_ENTRIES":
            if cohort_status not in {
                "WAITING_FOR_FIRST_20_RESOLUTIONS",
                "POST_ACTIVATION_COHORT_COMPLETE",
            }:
                return {
                    "status": "OVERRIDE_STATE_INVALID",
                    "resolved_value": fallback,
                    "error": "post-activation monitor gate returned an invalid status",
                }
            if cohort_status == "WAITING_FOR_FIRST_20_RESOLUTIONS" and (
                monitor_cohort.get("entry_count") != MIN_FORWARD_SAMPLE_COUNT
                or monitor_cohort.get("required_resolved_count")
                != MIN_FORWARD_SAMPLE_COUNT
                or isinstance(monitor_cohort.get("resolved_count"), bool)
                or not isinstance(monitor_cohort.get("resolved_count"), int)
                or not 0
                <= int(monitor_cohort.get("resolved_count"))
                < MIN_FORWARD_SAMPLE_COUNT
            ):
                return {
                    "status": "OVERRIDE_STATE_INVALID",
                    "resolved_value": fallback,
                    "error": "post-activation monitor resolution count is invalid",
                }
            if cohort_status == "POST_ACTIVATION_COHORT_COMPLETE" and (
                monitor_cohort.get("sample_count") != MIN_FORWARD_SAMPLE_COUNT
            ):
                return {
                    "status": "OVERRIDE_STATE_INVALID",
                    "resolved_value": fallback,
                    "error": "post-activation monitor sample count is invalid",
                }
            return {
                "status": "OVERRIDE_POST_ACTIVATION_MONITOR_PENDING",
                "resolved_value": max(float(fallback), candidate),
                "override": record,
                "post_activation_monitor": monitor_cohort,
            }
    return {
        "status": "ACTIVE_OVERRIDE",
        "resolved_value": max(float(fallback), candidate),
        "override": record,
    }


def resolve_forecast_confidence_floor_state(
    *,
    pair: str,
    method: str,
    lane_id: str | None = None,
    fallback: float,
    path: Path = DEFAULT_OVERRIDE_PATH,
    queue_path: Path = DEFAULT_WORK_ORDER_PATH,
) -> dict[str, Any]:
    """Resolve once per gateway phase; direct callers remain fully uncached."""

    cycle = _RUNTIME_VALIDATION_CACHE.get()
    key = (
        str(pair),
        str(method),
        str(lane_id or ""),
        float(fallback),
        str(path.resolve()),
        str(queue_path.resolve()),
    )
    generation: int | None = None
    if cycle is not None:
        with cycle.lock:
            if cycle.active:
                generation = cycle.generation
                cached = cycle.entries.get(key)
                if cached is not None:
                    return copy.deepcopy(cached)
    resolution = _resolve_forecast_confidence_floor_state_uncached(
        pair=pair,
        method=method,
        lane_id=lane_id,
        fallback=fallback,
        path=path,
        queue_path=queue_path,
    )
    if cycle is not None and generation is not None:
        with cycle.lock:
            # A final-pre-POST clear or owner exit may happen while the deep
            # attestation is in flight.  Such a result belongs to the old
            # generation and must never repopulate the fresh cache.
            if cycle.active and cycle.generation == generation:
                cycle.entries[key] = copy.deepcopy(resolution)
    return resolution


def resolve_forecast_confidence_floor(
    *,
    pair: str,
    method: str,
    lane_id: str | None = None,
    fallback: float,
    path: Path = DEFAULT_OVERRIDE_PATH,
    queue_path: Path = DEFAULT_WORK_ORDER_PATH,
) -> float:
    """Return the active evidence-bound tightening for a pair/method."""

    resolution = resolve_forecast_confidence_floor_state(
        pair=pair,
        method=method,
        lane_id=lane_id,
        fallback=fallback,
        path=path,
        queue_path=queue_path,
    )
    if resolution["status"] not in {"NO_OVERRIDE", "ACTIVE_OVERRIDE"}:
        raise ValueError(
            "guardian tuning override state is not safely resolvable: "
            + str(resolution["status"])
        )
    return float(resolution["resolved_value"])
