#!/usr/bin/env python3
"""Monitor the fixed first 20 entries after each activated tuning override."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.guardian_tuning_cohort import (  # noqa: E402
    build_post_activation_monitor_cohort,
)
from quant_rabbit.guardian_tuning_monitor import (  # noqa: E402
    seal_post_activation_monitor_evidence,
    validate_post_activation_monitor_evidence,
)
from quant_rabbit.guardian_tuning_overrides import (  # noqa: E402
    read_active_override_records,
    resolve_forecast_confidence_floor_state,
    runtime_forecast_floor_binding,
)
from tools.guardian_wake_dispatcher import (  # noqa: E402
    commit_tuning_override_monitor,
    reconcile_tuning_override_monitors,
)


MAX_MONITOR_EVIDENCE_FILES = 1_000


def _existing_valid_monitor_evidence(
    *,
    queue_path: Path,
    ledger_path: Path,
    record: dict[str, object],
) -> dict[str, object] | None:
    directory = queue_path.parent / "guardian_tuning_monitor_evidence"
    if not directory.exists():
        return None
    paths = sorted(directory.glob("*.json"))
    if len(paths) > MAX_MONITOR_EVIDENCE_FILES:
        raise ValueError("post-activation monitor evidence registry exceeds its bound")
    matches: list[dict[str, object]] = []
    for path in paths:
        if path.is_symlink() or re.fullmatch(r"[0-9a-f]{64}\.json", path.name) is None:
            raise ValueError("post-activation monitor evidence registry is invalid")
        digest = path.stem
        evidence_ref = (
            f"data/guardian_tuning_monitor_evidence/{path.name}"
            f"#sha256={digest}"
        )
        validation = validate_post_activation_monitor_evidence(
            queue_path=queue_path,
            ledger_path=ledger_path,
            evidence_ref=evidence_ref,
            expected_record=record,
        )
        status = str(validation.get("status") or "")
        if status == "MONITOR_EVIDENCE_ACTIVATION_CONFLICT":
            continue
        cohort_validation = validation.get("cohort_validation")
        current_truth_changed = (
            status == "MONITOR_EVIDENCE_COHORT_INVALID"
            and isinstance(cohort_validation, dict)
            and cohort_validation.get("status")
            == "POST_ACTIVATION_COHORT_CURRENT_TRUTH_CHANGED"
        )
        if current_truth_changed:
            # This is a valid artifact for the same activation whose frozen
            # first-20 truth was superseded by a late/backfilled broker fact.
            # Leave it immutable, but do not let it prevent sealing the newly
            # rebuilt canonical cohort on this retry.
            continue
        if status != "VALID":
            raise ValueError(
                "post-activation monitor evidence registry contains invalid current evidence: "
                + status
            )
        matches.append({"evidence_ref": evidence_ref, **validation})
    if not matches:
        return None
    decisions = {
        (str(item.get("decision") or ""), float(item.get("primary_metric_value")))
        for item in matches
    }
    if len(decisions) != 1:
        raise ValueError("post-activation monitor evidence registry is ambiguous")
    return matches[0]


def run_monitor(
    *,
    queue_path: Path,
    override_path: Path,
    ledger_path: Path,
    now: datetime,
) -> dict[str, object]:
    if ledger_path.resolve() != queue_path.with_name("execution_ledger.db").resolve():
        raise ValueError("post-activation monitor ledger must be the queue sibling")
    if override_path.resolve() != queue_path.with_name(
        "guardian_tuning_overrides.json"
    ).resolve():
        raise ValueError("post-activation monitor overrides must be the queue sibling")
    reconciliation = reconcile_tuning_override_monitors(
        path=queue_path,
        override_path=override_path,
        now=now,
    )
    if reconciliation.get("status") != "POST_ACTIVATION_MONITOR_RECONCILED":
        return {
            "status": "POST_ACTIVATION_MONITOR_FAILED",
            "reconciliation": reconciliation,
            "no_live_side_effects": True,
        }
    records = read_active_override_records(path=override_path)
    results: list[dict[str, object]] = []
    for record in records:
        override_key = str(record.get("override_key") or "")
        experiment_id = str(record.get("experiment_id") or "")
        lane_id = str(record.get("lane_id") or "")
        resolution = resolve_forecast_confidence_floor_state(
            pair=str(record.get("pair") or ""),
            method=str(record.get("method") or ""),
            lane_id=lane_id,
            fallback=float(record.get("previous_value")),
            path=override_path,
            queue_path=queue_path,
        )
        prior_decision = str(record.get("monitor_decision") or "")
        if prior_decision in {"KEEP", "QUARANTINE"}:
            expected_status = (
                "ACTIVE_OVERRIDE"
                if prior_decision == "KEEP"
                else "OVERRIDE_LANE_QUARANTINED"
            )
            if resolution.get("status") == expected_status:
                results.append(
                    {
                        "status": "POST_ACTIVATION_MONITOR_ALREADY_COMPLETE",
                        "override_key": override_key,
                        "decision": prior_decision,
                    }
                )
            else:
                results.append(
                    {
                        "status": "POST_ACTIVATION_MONITOR_OVERRIDE_NOT_ACTIVE",
                        "override_key": override_key,
                        "resolution_status": resolution.get("status"),
                        "retry_required": True,
                    }
                )
            continue
        if resolution.get("status") not in {
            "ACTIVE_OVERRIDE",
            "OVERRIDE_POST_ACTIVATION_MONITOR_PENDING",
        }:
            results.append(
                {
                    "status": "POST_ACTIVATION_MONITOR_OVERRIDE_NOT_ACTIVE",
                    "override_key": override_key,
                    "resolution_status": resolution.get("status"),
                    "retry_required": True,
                }
            )
            continue
        binding = runtime_forecast_floor_binding(
            lane_id=lane_id,
            override_path=override_path,
            queue_path=queue_path,
            allow_post_activation_monitor_pending=True,
        )
        if not math.isclose(
            float(binding["resolved_value"]),
            float(record.get("candidate_value")),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            results.append(
                {
                    "status": "POST_ACTIVATION_MONITOR_RUNTIME_BINDING_DRIFT",
                    "override_key": override_key,
                    "activated_candidate_value": record.get("candidate_value"),
                    "runtime_resolved_value": binding.get("resolved_value"),
                    "retry_required": True,
                }
            )
            continue
        cohort = build_post_activation_monitor_cohort(
            ledger_path=ledger_path,
            lane_id=lane_id,
            activated_at_utc=record.get("activated_at_utc"),
            activation_ledger_anchor=record.get("activation_ledger_anchor"),
        )
        if cohort.get("status") != "POST_ACTIVATION_COHORT_COMPLETE":
            results.append(
                {
                    "status": cohort.get("status"),
                    "override_key": override_key,
                    "cohort": cohort,
                    "retry_required": False,
                }
            )
            continue
        existing_evidence = _existing_valid_monitor_evidence(
            queue_path=queue_path,
            ledger_path=ledger_path,
            record=record,
        )
        if existing_evidence is None:
            sealed = seal_post_activation_monitor_evidence(
                project_root=queue_path.parent.parent,
                override_record=record,
                cohort=cohort,
                now=now,
            )
            evidence_ref = str(sealed["evidence_ref"])
            decision = str(sealed["payload"]["decision"])
            primary_metric_value = float(
                sealed["payload"]["primary_metric_value"]
            )
        else:
            evidence_ref = str(existing_evidence["evidence_ref"])
            decision = str(existing_evidence["decision"])
            primary_metric_value = float(
                existing_evidence["primary_metric_value"]
            )
        committed = commit_tuning_override_monitor(
            path=queue_path,
            override_path=override_path,
            override_key=override_key,
            experiment_id=experiment_id,
            monitor_evidence_ref=evidence_ref,
            decision=decision,
            primary_metric_value=primary_metric_value,
            now=now,
        )
        results.append({"override_key": override_key, **committed})
    failed = [
        item
        for item in results
        if item.get("status")
        not in {
            "POST_ACTIVATION_MONITOR_ALREADY_COMPLETE",
            "WAITING_FOR_FIRST_20_ENTRIES",
            "WAITING_FOR_FIRST_20_RESOLUTIONS",
            "POST_ACTIVATION_MONITOR_COMMITTED",
        }
    ]
    return {
        "status": (
            "POST_ACTIVATION_MONITOR_FAILED"
            if failed
            else "POST_ACTIVATION_MONITOR_OK"
        ),
        "active_override_count": len(records),
        "result_count": len(results),
        "results": results,
        "reconciliation": reconciliation,
        "no_live_side_effects": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue",
        type=Path,
        default=ROOT / "data" / "guardian_tuning_work_order.json",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        default=ROOT / "data" / "guardian_tuning_overrides.json",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "data" / "execution_ledger.db",
    )
    args = parser.parse_args(argv)
    try:
        result = run_monitor(
            queue_path=args.queue,
            override_path=args.overrides,
            ledger_path=args.ledger,
            now=datetime.now(timezone.utc),
        )
    except Exception as exc:  # fail closed at the operator boundary
        result = {
            "status": "POST_ACTIVATION_MONITOR_FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "no_live_side_effects": True,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") == "POST_ACTIVATION_MONITOR_OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
