#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools import guardian_wake_dispatcher as dispatcher  # noqa: E402
from quant_rabbit.guardian_tuning_evaluator import (  # noqa: E402
    EVALUATOR_NAME,
    FIXED_ACCEPTANCE_THRESHOLD,
    METRIC_NAMES,
    OBJECTIVE,
    PRIMARY_METRIC,
    SUPPORTED_THRESHOLD_PARAMETERS,
    evaluate_precommitted_threshold_cohort,
    source_identity,
    validate_source,
)
from quant_rabbit.guardian_tuning_cohort import (  # noqa: E402
    validate_canonical_forward_cohort,
)


APPROVED_EVALUATOR = ROOT / "tools" / "guardian_tuning_metric_evaluator.py"
RUNNER_NAME = "guardian_tuning_evidence_builder_v1"
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _current_work_order(path: Path, work_order_id: str) -> dict[str, Any]:
    loaded = dispatcher._load_tuning_work_order(path)
    if loaded.get("_read_error"):
        raise ValueError(str(loaded["_read_error"]))
    pending, _ = dispatcher._normalized_tuning_work_order_queue(loaded)
    matches = [
        item
        for item in pending
        if str(item.get("work_order_id") or "") == work_order_id
    ]
    if len(matches) != 1:
        raise ValueError("exactly one pending work order is required")
    return matches[0]


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def _write_immutable(path: Path, raw: bytes) -> None:
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"immutable artifact already exists with different bytes: {path}")
        return
    _atomic_bytes(path, raw)


def _relative_under(path: Path, root: Path, required: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(required.resolve())
    return resolved.relative_to(root.resolve())


def _snapshot_input(source: Path, *, kind: str) -> str:
    raw = source.resolve(strict=True).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    suffix = source.suffix if source.suffix else ".bin"
    relative = (
        Path("data")
        / "guardian_tuning_experiment_inputs"
        / kind
        / f"{digest}{suffix}"
    )
    target = ROOT / relative
    _write_immutable(target, raw)
    return f"{relative}#sha256={digest}"


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _source_identity(path: Path) -> tuple[str, Any, int, str]:
    payload = _read_json_object(path)
    return source_identity(payload)


def _artifact_path(
    *, queue_path: Path, artifact_ref: str, allowed_root: str
) -> tuple[Path, str]:
    validation = dispatcher._validate_project_artifact_ref(
        queue_path=queue_path,
        artifact_ref=artifact_ref,
        allowed_roots=(allowed_root,),
    )
    if validation.get("status") != "VALID":
        raise ValueError(f"invalid immutable artifact reference: {validation}")
    return Path(str(validation["path"])), str(validation["sha256"])


def _prepared_contract(
    path: Path,
    *,
    work_order_id: str,
    observation_id: str,
    experiment_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    work_order = _current_work_order(path, work_order_id)
    latest_observation = str(
        work_order.get("latest_observation_id")
        or work_order.get("observation_id")
        or work_order.get("event_fingerprint")
        or ""
    )
    if latest_observation != observation_id:
        raise ValueError("expected observation is stale")
    contract = work_order.get("prepared_experiment_contract")
    if not isinstance(contract, dict):
        raise ValueError("experiment contract has not been prepared")
    if (
        contract.get("status") != "PREPARED"
        or str(contract.get("observation_id") or "") != observation_id
        or str(contract.get("experiment_id") or "") != experiment_id
    ):
        raise ValueError("prepared experiment identity does not match")
    return work_order, contract


def _init_run(args: argparse.Namespace) -> dict[str, Any]:
    if _SAFE_ID.fullmatch(args.experiment_id) is None:
        raise ValueError("experiment-id contains unsupported characters")
    if args.evaluator_artifact.resolve(strict=True) != APPROVED_EVALUATOR.resolve(strict=True):
        raise ValueError("only the approved read-only metric evaluator may be prepared")
    source_path = args.source_data.resolve(strict=True)
    source_path.relative_to((ROOT / "data" / "guardian_tuning_cohorts").resolve())
    source_raw = source_path.read_bytes()
    if source_path.stem != hashlib.sha256(source_raw).hexdigest():
        raise ValueError("source-data must be a content-addressed cohort-builder artifact")
    source_watermark = json.loads(args.source_watermark_json)
    if not isinstance(source_watermark, dict) or not source_watermark:
        raise ValueError("source-watermark-json must be a non-empty JSON object")
    source_payload = _read_json_object(args.source_data)
    source_contract = validate_source(source_payload)
    source_cohort, source_mark, source_count, source_parameter = source_identity(
        source_payload
    )
    if (
        source_cohort != args.cohort_id
        or source_mark != source_watermark
        or source_count != args.sample_count
    ):
        raise ValueError("precommitted cohort identity does not match frozen source data")
    work_order = _current_work_order(args.path, args.work_order_id)
    review = work_order.get("bot_tuning_review")
    adjustments = review.get("proposed_adjustments") if isinstance(review, dict) else None
    if not isinstance(adjustments, list) or len(adjustments) != 1:
        raise ValueError("one TEST_REQUIRED adjustment must be bound before preparation")
    canonical_validation = validate_canonical_forward_cohort(
        source_payload,
        ledger_path=ROOT / "data" / "execution_ledger.db",
        entry_thesis_path=ROOT / "data" / "entry_thesis_ledger.jsonl",
        forecast_history_path=ROOT / "data" / "forecast_history.jsonl",
        review=review,
    )
    if canonical_validation.get("status") != "VALID":
        raise ValueError(
            f"source cohort is not canonical forward evidence: {canonical_validation}"
        )
    adjustment = adjustments[0]
    review_parameter = str(adjustment.get("parameter") or "")
    if review_parameter != source_parameter or review_parameter not in SUPPORTED_THRESHOLD_PARAMETERS:
        raise ValueError("review parameter is unsupported or does not match frozen source data")
    if (
        source_contract["pair"] != str(adjustment.get("pair") or "").upper()
        or source_contract["bot_family"]
        != str(adjustment.get("bot_family") or "").lower()
    ):
        raise ValueError("frozen cohort pair/family does not match the reviewed adjustment")
    metric_names = sorted(METRIC_NAMES)
    if args.primary_metric != PRIMARY_METRIC or args.objective != OBJECTIVE:
        raise ValueError("evaluator v1 requires its fixed primary metric and objective")
    if args.acceptance_threshold != FIXED_ACCEPTANCE_THRESHOLD:
        raise ValueError("evaluator v1 uses its fixed strict-improvement threshold")

    source_ref = _snapshot_input(args.source_data, kind="data")
    evaluator_ref = _snapshot_input(args.evaluator_artifact, kind="evaluators")
    now = datetime.now(timezone.utc)
    prepared = dispatcher.prepare_tuning_experiment_contract(
        path=args.path,
        work_order_id=args.work_order_id,
        expected_observation_id=args.expected_observation_id,
        experiment_id=args.experiment_id,
        cohort_id=args.cohort_id,
        source_watermark=source_watermark,
        sample_count=args.sample_count,
        evaluator=args.evaluator,
        source_data_ref=source_ref,
        evaluator_artifact_ref=evaluator_ref,
        primary_metric=args.primary_metric,
        objective=args.objective,
        acceptance_threshold=args.acceptance_threshold,
        metric_names=metric_names,
        prepared_by=args.prepared_by,
        now=now,
    )
    if prepared.get("status") not in {
        "EXPERIMENT_CONTRACT_PREPARED",
        "EXPERIMENT_CONTRACT_ALREADY_PREPARED",
    }:
        return prepared
    contract = prepared["prepared_experiment_contract"]
    output = args.output or (
        ROOT
        / "data"
        / "guardian_tuning_experiment_runs"
        / f"{args.experiment_id}.pending.json"
    )
    _relative_under(output, ROOT, ROOT / "data" / "guardian_tuning_experiment_runs")
    payload = {
        "schema_version": 1,
        "status": "PREPARED",
        "work_order_id": args.work_order_id,
        "observation_id": args.expected_observation_id,
        "experiment_id": args.experiment_id,
        "experiment_contract_digest": contract["experiment_contract_digest"],
        "cohort_id": contract["cohort_id"],
        "source_watermark": contract["source_watermark"],
        "sample_count": contract["sample_count"],
        "evaluator": contract["evaluator"],
        "source_data_ref": contract["source_data_ref"],
        "evaluator_artifact_ref": contract["evaluator_artifact_ref"],
        "primary_metric": contract["primary_metric"],
        "objective": contract["objective"],
        "acceptance_threshold": contract["acceptance_threshold"],
        "metric_names": contract["metric_names"],
        "prepared_at_utc": contract["prepared_at_utc"],
        "no_live_side_effects": True,
    }
    _write_immutable(output, _json_bytes(payload))
    return {
        "status": "EXPERIMENT_RUN_TEMPLATE_WRITTEN",
        "path": str(output),
        "experiment_contract_digest": contract["experiment_contract_digest"],
    }


def _content_addressed_json(
    payload: dict[str, Any], *, required_root: Path, output: Path | None = None
) -> tuple[Path, str, str]:
    raw = _json_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest()
    target = required_root / f"{digest}.json"
    if output is not None and output.resolve() != target.resolve():
        raise ValueError(f"output must be the content-addressed path {target}")
    _write_immutable(target, raw)
    relative = target.resolve().relative_to(ROOT.resolve())
    return target, str(relative), digest


def _seal_source(
    *,
    queue_path: Path,
    source_path: Path,
    output: Path | None,
) -> dict[str, Any]:
    source_payload = _read_json_object(source_path)
    raw_source = source_path.read_bytes()
    source_digest = hashlib.sha256(raw_source).hexdigest()
    if source_path.name != f"{source_digest}.json":
        raise ValueError("completed run must already use its content-addressed filename")
    source_relative = _relative_under(
        source_path,
        ROOT,
        ROOT / "data" / "guardian_tuning_experiment_runs",
    )
    work_order_id = str(source_payload.get("work_order_id") or "")
    observation_id = str(source_payload.get("observation_id") or "")
    experiment_id = str(source_payload.get("experiment_id") or "")
    result = str(source_payload.get("result") or "")
    work_order, contract = _prepared_contract(
        queue_path,
        work_order_id=work_order_id,
        observation_id=observation_id,
        experiment_id=experiment_id,
    )
    review = work_order.get("bot_tuning_review")
    if not isinstance(review, dict):
        raise ValueError("current work order review is missing")
    adjustments = review.get("proposed_adjustments")
    if not isinstance(adjustments, list) or len(adjustments) != 1:
        raise ValueError("current work order must have exactly one adjustment")
    adjustment = adjustments[0]
    execution = source_payload.get("evaluator_execution")
    if not isinstance(execution, dict) or execution.get("runner") != RUNNER_NAME:
        raise ValueError("completed run lacks builder execution provenance")
    source_ref = f"{source_relative}#sha256={source_digest}"
    validation = dispatcher._validate_tuning_experiment_run_ref(
        queue_path=queue_path,
        source_artifact_ref=source_ref,
        work_order_id=work_order_id,
        observation_id=observation_id,
        experiment_id=experiment_id,
        experiment_result=result,
        review=review,
        semantic_state_id=dispatcher._work_order_semantic_state_id(work_order),
        prepared_contract=contract,
        work_order_generated_at=work_order.get("generated_at_utc"),
        review_completed_at_utc=work_order.get(
            "structured_review_completed_at_utc"
        ),
        now=datetime.now(timezone.utc),
    )
    if validation.get("status") != "VALID":
        return {"status": "SOURCE_EXPERIMENT_INVALID", "validation": validation}
    evidence = {
        "schema_version": 1,
        "status": "COMPLETED",
        "work_order_id": work_order_id,
        "observation_id": observation_id,
        "experiment_id": experiment_id,
        "review_digest_sha256": dispatcher._tuning_review_digest(review),
        "hypothesis": review.get("hypothesis"),
        "falsifiable_experiment": review.get("falsifiable_experiment"),
        "pair": adjustment.get("pair"),
        "bot_family": adjustment.get("bot_family"),
        "parameter": adjustment.get("parameter"),
        "current_value": adjustment.get("current_value"),
        "candidate_value": adjustment.get("candidate_value"),
        "result": result,
        "source_artifact_ref": source_ref,
        "experiment_contract_digest": validation.get("experiment_contract_digest"),
        "generated_at_utc": source_payload.get("generated_at_utc"),
        "no_live_side_effects": True,
    }
    evidence_path, evidence_relative, evidence_digest = _content_addressed_json(
        evidence,
        required_root=ROOT / "data" / "guardian_tuning_evidence",
        output=output,
    )
    return {
        "status": "EXPERIMENT_EVIDENCE_SEALED",
        "path": str(evidence_path),
        "experiment_evidence_ref": f"{evidence_relative}#sha256={evidence_digest}",
        "experiment_contract_digest": validation.get("experiment_contract_digest"),
        "experiment_result": result,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    work_order, contract = _prepared_contract(
        args.path,
        work_order_id=args.work_order_id,
        observation_id=args.expected_observation_id,
        experiment_id=args.experiment_id,
    )
    review = work_order.get("bot_tuning_review")
    if not isinstance(review, dict):
        raise ValueError("current work order review is missing")
    adjustments = review.get("proposed_adjustments")
    if not isinstance(adjustments, list) or len(adjustments) != 1:
        raise ValueError("current work order must have exactly one adjustment")
    adjustment = adjustments[0]
    source_path, source_sha = _artifact_path(
        queue_path=args.path,
        artifact_ref=str(contract["source_data_ref"]),
        allowed_root="data/guardian_tuning_experiment_inputs/data",
    )
    _, evaluator_sha = _artifact_path(
        queue_path=args.path,
        artifact_ref=str(contract["evaluator_artifact_ref"]),
        allowed_root="data/guardian_tuning_experiment_inputs/evaluators",
    )
    if contract.get("evaluator") != EVALUATOR_NAME:
        raise ValueError("prepared evaluator identity is not approved")
    approved_sha = hashlib.sha256(APPROVED_EVALUATOR.read_bytes()).hexdigest()
    if evaluator_sha != approved_sha:
        raise ValueError("prepared evaluator bytes do not match the approved evaluator")

    source_raw, source_error = dispatcher._bounded_file_bytes(
        source_path,
        max_bytes=dispatcher.MAX_TUNING_SOURCE_BYTES,
    )
    if source_error is not None or source_raw is None:
        raise ValueError(source_error or "frozen source read failed")
    if hashlib.sha256(source_raw).hexdigest() != source_sha:
        raise ValueError("frozen source changed after content-address validation")
    source_payload = json.loads(source_raw.decode("utf-8"))
    if not isinstance(source_payload, dict):
        raise ValueError("frozen source must be a JSON object")
    evaluation = evaluate_precommitted_threshold_cohort(
        source_payload,
        parameter=str(adjustment.get("parameter") or ""),
        current_value=adjustment.get("current_value"),
        candidate_value=adjustment.get("candidate_value"),
        primary_metric=str(contract["primary_metric"]),
        objective=str(contract["objective"]),
        acceptance_threshold=contract["acceptance_threshold"],
    )
    stdout_raw = (
        json.dumps(evaluation, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    stderr_raw = b""
    if not isinstance(evaluation, dict) or evaluation.get("status") != "EVALUATION_COMPLETED":
        raise ValueError("trusted evaluator did not return a completed result")
    expected_evaluation = {
        "evaluator": EVALUATOR_NAME,
        "cohort_id": contract["cohort_id"],
        "source_watermark": contract["source_watermark"],
        "sample_count": contract["sample_count"],
        "parameter": adjustment.get("parameter"),
        "current_value": float(adjustment.get("current_value")),
        "candidate_value": float(adjustment.get("candidate_value")),
        "metric_names": contract["metric_names"],
        "primary_metric": contract["primary_metric"],
        "objective": contract["objective"],
        "acceptance_threshold": contract["acceptance_threshold"],
    }
    mismatched = [
        key for key, expected in expected_evaluation.items() if evaluation.get(key) != expected
    ]
    if mismatched:
        raise ValueError(f"evaluator output conflicts with prepared contract: {mismatched}")
    result = str(evaluation.get("derived_result") or "")
    if result not in {"ACCEPTED_IMPROVEMENT", "REJECTED_NO_IMPROVEMENT"}:
        raise ValueError("approved evaluator did not derive a supported result")
    generated_at = datetime.now(timezone.utc).isoformat()
    run_payload = {
        "schema_version": 1,
        "status": "COMPLETED",
        "exit_status": (
            "COMPLETED_SUCCESS"
            if result == "ACCEPTED_IMPROVEMENT"
            else "COMPLETED_NO_EDGE"
        ),
        "work_order_id": args.work_order_id,
        "observation_id": args.expected_observation_id,
        "experiment_id": args.experiment_id,
        "experiment_contract_digest": contract["experiment_contract_digest"],
        "review_digest_sha256": dispatcher._tuning_review_digest(review),
        "pair": adjustment.get("pair"),
        "bot_family": adjustment.get("bot_family"),
        "parameter": adjustment.get("parameter"),
        "current_value": adjustment.get("current_value"),
        "candidate_value": adjustment.get("candidate_value"),
        "cohort_id": evaluation["cohort_id"],
        "source_watermark": evaluation["source_watermark"],
        "sample_count": evaluation["sample_count"],
        "baseline_metrics": evaluation["baseline_metrics"],
        "candidate_metrics": evaluation["candidate_metrics"],
        "acceptance_constraints": evaluation["acceptance_constraints"],
        "evaluator": contract["evaluator"],
        "source_data_ref": contract["source_data_ref"],
        "evaluator_artifact_ref": contract["evaluator_artifact_ref"],
        "primary_metric": contract["primary_metric"],
        "objective": contract["objective"],
        "acceptance_threshold": contract["acceptance_threshold"],
        "result": result,
        "generated_at_utc": generated_at,
        "evaluator_execution": {
            "runner": RUNNER_NAME,
            "exit_code": 0,
            "stdout_sha256": hashlib.sha256(stdout_raw).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr_raw).hexdigest(),
            "source_data_sha256": source_sha,
            "evaluator_artifact_sha256": evaluator_sha,
            "executed_at_utc": generated_at,
        },
        "no_live_side_effects": True,
    }
    run_path, _, _ = _content_addressed_json(
        run_payload,
        required_root=ROOT / "data" / "guardian_tuning_experiment_runs",
    )
    return _seal_source(queue_path=args.path, source_path=run_path, output=args.output)


def _abort(args: argparse.Namespace) -> dict[str, Any]:
    _, contract = _prepared_contract(
        args.path,
        work_order_id=args.work_order_id,
        observation_id=args.expected_observation_id,
        experiment_id=args.experiment_id,
    )
    now = datetime.now(timezone.utc)
    failure = {
        "schema_version": 1,
        "status": "ABORTED",
        "work_order_id": args.work_order_id,
        "observation_id": args.expected_observation_id,
        "experiment_id": args.experiment_id,
        "experiment_contract_digest": contract["experiment_contract_digest"],
        "experiment_semantic_digest": contract["experiment_semantic_digest"],
        "aborted_by": args.aborted_by,
        "reason": args.reason,
        "generated_at_utc": now.isoformat(),
        "no_live_side_effects": True,
    }
    failure_path, failure_relative, failure_digest = _content_addressed_json(
        failure,
        required_root=ROOT / "data" / "guardian_tuning_experiment_failures",
    )
    failure_ref = f"{failure_relative}#sha256={failure_digest}"
    result = dispatcher.abort_tuning_experiment_contract(
        path=args.path,
        work_order_id=args.work_order_id,
        expected_observation_id=args.expected_observation_id,
        experiment_id=args.experiment_id,
        aborted_by=args.aborted_by,
        reason=args.reason,
        failure_evidence_ref=failure_ref,
        now=now,
    )
    return {
        **result,
        "failure_evidence_path": str(failure_path),
        "failure_evidence_ref": failure_ref,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Precommit, execute, and seal one read-only guardian tuning experiment."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=ROOT / "data" / "guardian_tuning_work_order.json",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init-run")
    init.add_argument("--work-order-id", required=True)
    init.add_argument("--expected-observation-id", required=True)
    init.add_argument("--experiment-id", required=True)
    init.add_argument("--source-data", type=Path, required=True)
    init.add_argument(
        "--evaluator-artifact",
        type=Path,
        default=APPROVED_EVALUATOR,
    )
    init.add_argument("--cohort-id", required=True)
    init.add_argument("--source-watermark-json", required=True)
    init.add_argument("--sample-count", type=int, required=True)
    init.add_argument("--evaluator", choices=(EVALUATOR_NAME,), default=EVALUATOR_NAME)
    init.add_argument("--primary-metric", choices=(PRIMARY_METRIC,), default=PRIMARY_METRIC)
    init.add_argument("--objective", choices=(OBJECTIVE,), default=OBJECTIVE)
    init.add_argument(
        "--acceptance-threshold",
        type=float,
        choices=(FIXED_ACCEPTANCE_THRESHOLD,),
        default=FIXED_ACCEPTANCE_THRESHOLD,
    )
    init.add_argument("--prepared-by", required=True)
    init.add_argument("--output", type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("--work-order-id", required=True)
    run.add_argument("--expected-observation-id", required=True)
    run.add_argument("--experiment-id", required=True)
    run.add_argument("--output", type=Path)
    abort = subparsers.add_parser("abort")
    abort.add_argument("--work-order-id", required=True)
    abort.add_argument("--expected-observation-id", required=True)
    abort.add_argument("--experiment-id", required=True)
    abort.add_argument("--aborted-by", required=True)
    abort.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "init-run":
            result = _init_run(args)
        elif args.command == "run":
            result = _run(args)
        else:
            result = _abort(args)
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result = {"status": "EVIDENCE_BUILDER_FAILED", "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") in {
        "EXPERIMENT_RUN_TEMPLATE_WRITTEN",
        "EXPERIMENT_EVIDENCE_SEALED",
        "EXPERIMENT_CONTRACT_ABORTED",
        "EXPERIMENT_CONTRACT_ALREADY_ABORTED",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
