"""Canonical first-20 post-activation monitoring for guardian tuning."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.guardian_tuning_cohort import (
    validate_post_activation_monitor_cohort,
)


MAX_MONITOR_EVIDENCE_BYTES = 512 * 1024
MONITOR_EVIDENCE_DIRECTORY = "guardian_tuning_monitor_evidence"
PRIMARY_METRIC = "net_jpy_per_1000_units_per_opportunity"


def _canonical_raw(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("post-activation monitor metric must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("post-activation monitor metric must be finite")
    return parsed


def _same_finite_number(left: object, right: object) -> bool:
    try:
        return _finite(left) == _finite(right)
    except (OverflowError, TypeError, ValueError):
        return False


def seal_post_activation_monitor_evidence(
    *,
    project_root: Path,
    override_record: dict[str, Any],
    cohort: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    """Seal one immutable KEEP/QUARANTINE decision from the fixed cohort."""

    if cohort.get("status") != "POST_ACTIVATION_COHORT_COMPLETE":
        raise ValueError("post-activation monitor cohort is not complete")
    metric = _finite(cohort.get("primary_metric_value"))
    lane_id = str(override_record.get("lane_id") or "")
    if (
        not lane_id
        or not isinstance(override_record.get("activation_ledger_anchor"), dict)
        or override_record["activation_ledger_anchor"].get("captured_at_utc")
        != override_record.get("activated_at_utc")
        or cohort.get("lane_id") != lane_id
        or cohort.get("activated_at_utc")
        != override_record.get("activated_at_utc")
        or cohort.get("activation_ledger_anchor")
        != override_record.get("activation_ledger_anchor")
        or int(cohort.get("sample_count") or 0) != 20
    ):
        raise ValueError("post-activation monitor cohort identity conflicts")
    decision = "KEEP" if metric > 0.0 else "QUARANTINE"
    payload = {
        "schema_version": 1,
        "status": "POST_ACTIVATION_MONITOR_COMPLETED",
        "decision": decision,
        "override_key": override_record.get("override_key"),
        "work_order_id": override_record.get("work_order_id"),
        "experiment_id": override_record.get("experiment_id"),
        "activation_manifest_ref": override_record.get("activation_manifest_ref"),
        "terminal_confirmation_sha256": override_record.get(
            "terminal_confirmation_sha256"
        ),
        "lane_id": lane_id,
        "pair": override_record.get("pair"),
        "method": override_record.get("method"),
        "parameter": override_record.get("parameter"),
        "candidate_value": override_record.get("candidate_value"),
        "activated_at_utc": override_record.get("activated_at_utc"),
        "activation_ledger_anchor": override_record.get("activation_ledger_anchor"),
        "cohort_id": cohort.get("cohort_id"),
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_value": metric,
        "sample_count": 20,
        "cohort": cohort,
        "generated_at_utc": now.astimezone(timezone.utc).isoformat(),
        "live_permission_allowed": False,
        "no_direct_oanda": True,
    }
    raw = _canonical_raw(payload)
    if len(raw) > MAX_MONITOR_EVIDENCE_BYTES:
        raise ValueError("post-activation monitor evidence exceeds its bound")
    digest = hashlib.sha256(raw).hexdigest()
    directory = project_root / "data" / MONITOR_EVIDENCE_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{digest}.json"
    try:
        with destination.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if destination.read_bytes() != raw:
            raise ValueError("post-activation monitor content address conflicts") from None
    ref = f"data/{MONITOR_EVIDENCE_DIRECTORY}/{digest}.json#sha256={digest}"
    return {"status": "MONITOR_EVIDENCE_SEALED", "evidence_ref": ref, "payload": payload}


def validate_post_activation_monitor_evidence(
    *,
    queue_path: Path,
    ledger_path: Path,
    evidence_ref: object,
    expected_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate content address, exact activation identity, and ledger prefix."""

    match = re.fullmatch(
        rf"data/{MONITOR_EVIDENCE_DIRECTORY}/([0-9a-f]{{64}})\.json"
        r"#sha256=([0-9a-f]{64})",
        str(evidence_ref or ""),
    )
    if match is None or match.group(1) != match.group(2):
        return {"status": "MONITOR_EVIDENCE_REF_INVALID"}
    digest = match.group(1)
    project_root = queue_path.parent.parent.resolve()
    path = project_root / "data" / MONITOR_EVIDENCE_DIRECTORY / f"{digest}.json"
    try:
        if path.is_symlink():
            raise ValueError("monitor evidence must not be a symlink")
        raw = path.read_bytes()
    except (OSError, ValueError) as exc:
        return {
            "status": "MONITOR_EVIDENCE_READ_FAILED",
            "error": f"{type(exc).__name__}: {exc}",
        }
    if (
        len(raw) > MAX_MONITOR_EVIDENCE_BYTES
        or hashlib.sha256(raw).hexdigest() != digest
    ):
        return {"status": "MONITOR_EVIDENCE_CONTENT_ADDRESS_INVALID"}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        return {"status": "MONITOR_EVIDENCE_JSON_INVALID"}
    if not isinstance(payload, dict):
        return {"status": "MONITOR_EVIDENCE_SCHEMA_INVALID"}
    try:
        outer_metric = _finite(payload.get("primary_metric_value"))
    except (OverflowError, TypeError, ValueError):
        return {"status": "MONITOR_EVIDENCE_METRIC_INVALID"}
    decision = str(payload.get("decision") or "")
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "POST_ACTIVATION_MONITOR_COMPLETED"
        or decision not in {"KEEP", "QUARANTINE"}
        or type(payload.get("sample_count")) is not int
        or payload.get("sample_count") != 20
        or payload.get("primary_metric") != PRIMARY_METRIC
        or not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("cohort_id") or ""))
        or not isinstance(payload.get("activation_ledger_anchor"), dict)
        or payload.get("live_permission_allowed") is not False
        or payload.get("no_direct_oanda") is not True
        or not isinstance(payload.get("cohort"), dict)
    ):
        return {"status": "MONITOR_EVIDENCE_SCHEMA_INVALID"}
    if expected_record is not None:
        fields = (
            "override_key",
            "work_order_id",
            "experiment_id",
            "activation_manifest_ref",
            "terminal_confirmation_sha256",
            "lane_id",
            "pair",
            "method",
            "parameter",
            "candidate_value",
            "activated_at_utc",
            "activation_ledger_anchor",
        )
        if any(payload.get(field) != expected_record.get(field) for field in fields):
            return {"status": "MONITOR_EVIDENCE_ACTIVATION_CONFLICT"}
    cohort_validation = validate_post_activation_monitor_cohort(
        payload["cohort"],
        ledger_path=ledger_path,
    )
    if cohort_validation.get("status") != "VALID":
        return {
            "status": "MONITOR_EVIDENCE_COHORT_INVALID",
            "cohort_validation": cohort_validation,
        }
    cohort = payload["cohort"]
    identity_fields = (
        "lane_id",
        "pair",
        "method",
        "activated_at_utc",
        "activation_ledger_anchor",
    )
    if (
        any(cohort.get(field) != payload.get(field) for field in identity_fields)
        or payload["activation_ledger_anchor"].get("captured_at_utc")
        != payload.get("activated_at_utc")
        or cohort.get("cohort_id") != payload.get("cohort_id")
        or cohort_validation.get("cohort_id") != payload.get("cohort_id")
        or not isinstance(cohort.get("activation_ledger_anchor"), dict)
        or type(cohort.get("sample_count")) is not int
        or cohort.get("sample_count") != payload.get("sample_count")
        or cohort.get("primary_metric") != payload.get("primary_metric")
        or not _same_finite_number(
            cohort.get("primary_metric_value"),
            outer_metric,
        )
        or not _same_finite_number(
            cohort_validation.get("primary_metric_value"),
            outer_metric,
        )
    ):
        return {
            "status": "MONITOR_EVIDENCE_COHORT_BINDING_INVALID",
            "cohort_validation": cohort_validation,
        }
    validated_metric = _finite(cohort_validation.get("primary_metric_value"))
    validated_decision = "KEEP" if validated_metric > 0.0 else "QUARANTINE"
    if decision != validated_decision:
        return {
            "status": "MONITOR_EVIDENCE_DECISION_INVALID",
            "validated_decision": validated_decision,
        }
    return {
        "status": "VALID",
        "decision": validated_decision,
        "primary_metric_value": validated_metric,
        "payload": payload,
        "sha256": digest,
    }
