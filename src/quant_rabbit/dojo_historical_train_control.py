"""Reviewed, research-only controller for continuous DOJO historical TRAIN.

The controller closes the gap between the sealed long-horizon primitives and
an actual replay process.  It accepts one repository-owned run-control policy,
materializes one observed-only month slice, executes one immutable job, then
publishes the complete coordinate denominator and carry state.  It has no
broker client, credential input, live permission, promotion path, or model
client.

Generation mutation is deliberately outside this module.  Economic results
become trainer input only after both intrabar paths for a fixed six-month M5
block are terminal.  A trainer response must create a new immutable generation;
it cannot modify the running G2 schedule.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_bot_trainer import (
    PROPOSAL_CONTRACT,
    seal_candidate_proposal,
)
from quant_rabbit.dojo_long_horizon_economic_runner import (
    DojoLongHorizonEconomicRunnerError,
    ECONOMIC_JOB_RESULT_CONTRACT,
    build_sparse_month_source_slice_receipt,
    run_long_horizon_economic_job,
    validate_sparse_month_source_slice_receipt,
)
from quant_rabbit.dojo_long_horizon_execution import (
    CELL_CONTRACT,
    build_long_horizon_coordinate_result,
    claim_next_long_horizon_job,
    initialize_long_horizon_execution_state,
    long_horizon_execution_status,
    record_long_horizon_coordinate_results,
    seal_long_horizon_attempt,
)
from quant_rabbit.dojo_long_horizon_plan import (
    IMPLEMENTATION_DIGEST_KEYS,
    RAPID_2025H1_PROFILE,
    build_long_horizon_train_plan,
    canonical_sha256,
    validate_long_horizon_train_plan,
)
from quant_rabbit.dojo_long_horizon_schedule import (
    build_long_horizon_stream_schedule,
    validate_long_horizon_stream_schedule,
)
from quant_rabbit.dojo_long_horizon_source_manifest import (
    long_horizon_plan_digest_inputs,
    verify_long_horizon_source_manifest_seal,
)
from quant_rabbit.dojo_portfolio_replay_reducer import seal_portfolio_policy
from quant_rabbit.dojo_sparse_replay import SparseReplayError
from quant_rabbit.dojo_tuned_strategy_runtime import (
    build_tuned_strategy_runtime_factory,
    build_tuned_strategy_runtime_seal,
    verify_tuned_strategy_runtime_seal,
)


CONTROL_CONTRACT: Final = "QR_DOJO_G2_HISTORICAL_RUN_CONTROL_V1"
MANIFEST_CONTRACT: Final = "QR_DOJO_HISTORICAL_TRAIN_GENERATION_MANIFEST_V1"
MILESTONE_CONTRACT: Final = "QR_DOJO_HISTORICAL_TRAIN_MILESTONE_V1"
JOB_COMPLETION_CONTRACT: Final = "QR_DOJO_HISTORICAL_TRAIN_JOB_COMPLETION_V1"
SCHEMA_VERSION: Final = 1
GENERATION_ORDINAL: Final = 2
RUNNER_ID: Final = "dojo-g2-historical-train-v1"
MAX_JSON_BYTES: Final = 256 * 1024 * 1024
NON_JPY_STRESS_FINANCING_JPY_PER_UNIT_DAY: Final = 0.02

_AUTHORITY: Final = {
    "automatic_deployment_allowed": False,
    "broker_mutation_allowed": False,
    "live_permission": False,
    "order_authority": "NONE",
    "promotion_eligible": False,
}

_ARTIFACT_NAMES: Final = {
    "control_manifest": "control-manifest.json",
    "plan": "plan.json",
    "proposals": "candidate-proposals.json",
    "resource_policy": "resource-policy.json",
    "runtime_seal": "tuned-runtime-seal.json",
    "schedule": "schedule.json",
    "worker_catalog": "worker-catalog.json",
}


class DojoHistoricalTrainControlError(ValueError):
    """A sealed input, resource gate, or immutable job transition is unsafe."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoHistoricalTrainControlError(
            "artifact is not strict canonical JSON"
        ) from exc


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    candidate = Path(path)
    try:
        before = candidate.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalTrainControlError(f"{field} is unavailable") from exc
    if (
        candidate.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or not 0 < before.st_size <= MAX_JSON_BYTES
    ):
        raise DojoHistoricalTrainControlError(
            f"{field} must be a bounded nonempty regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(candidate, flags)
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        raw = handle.read(MAX_JSON_BYTES + 1)
        opened = os.fstat(handle.fileno())
    current = candidate.stat(follow_symlinks=False)
    identities = {
        (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
        for row in (before, opened, current)
    }
    if len(identities) != 1 or len(raw) != before.st_size:
        raise DojoHistoricalTrainControlError(f"{field} changed while read")

    def reject_constant(token: str) -> None:
        raise DojoHistoricalTrainControlError(
            f"{field} contains non-finite JSON: {token}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise DojoHistoricalTrainControlError(
                    f"{field} contains duplicate key: {key}"
                )
            result[key] = item
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except DojoHistoricalTrainControlError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoHistoricalTrainControlError(f"{field} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise DojoHistoricalTrainControlError(f"{field} must be a JSON object")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise DojoHistoricalTrainControlError(
                f"immutable artifact already exists with different bytes: {path}"
            )
        return
    temporary = path.with_name(
        f".{path.name}.{hashlib.sha256(payload).hexdigest()}.{os.getpid()}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            if handle.write(payload) != len(payload):
                raise DojoHistoricalTrainControlError(
                    "immutable artifact write was incomplete"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _file_binding(repo_root: Path, relative_paths: Sequence[str]) -> str:
    rows = []
    for relative in relative_paths:
        path = repo_root / relative
        state = path.stat(follow_symlinks=False)
        if path.is_symlink() or not stat.S_ISREG(state.st_mode) or state.st_size <= 0:
            raise DojoHistoricalTrainControlError(
                f"implementation dependency is invalid: {relative}"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {"relative_path": relative, "size_bytes": state.st_size, "sha256": digest}
        )
    return canonical_sha256(rows)


def _verified_control(
    run_control_path: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    control = _read_json(run_control_path, field="run control")
    if control.get("contract") != CONTROL_CONTRACT or control.get("schema_version") != 1:
        raise DojoHistoricalTrainControlError("run-control contract/version drifted")
    authority = control.get("authority")
    if not isinstance(authority, Mapping) or (
        authority.get("historical_replay_process_start_allowed") is not True
        or authority.get("research_filesystem_write_allowed") is not True
        or authority.get("broker_mutation_allowed") is not False
        or authority.get("live_permission") is not False
        or authority.get("order_authority") != "NONE"
    ):
        raise DojoHistoricalTrainControlError("run-control authority is unsafe")
    fixed = control.get("fixed_inputs")
    execution = control.get("execution")
    milestone = control.get("trainer_milestones")
    if not all(isinstance(row, Mapping) for row in (fixed, execution, milestone)):
        raise DojoHistoricalTrainControlError("run-control sections are malformed")
    if (
        fixed.get("generation") != "G2"
        or fixed.get("study_profile") != RAPID_2025H1_PROFILE
        or execution.get("evidence_tier") != "WORN_HISTORICAL_TRAIN_ONLY"
        or execution.get("max_parallel_jobs") != 1
        or execution.get("max_jobs_per_invocation") != 1
        or milestone.get("m5_completed_months_per_review") != 6
        or milestone.get("minimum_completed_intrabar_paths_per_month") != 2
        or milestone.get("rapid_train_evaluation_mode") != "INDEPENDENT_MONTH"
        or milestone.get("continuous_account_role")
        != "SEPARATE_LONG_PROFILE_NOT_RAPID_TUNING_GATE"
        or milestone.get("non_overlapping_six_month_blocks_required") is not True
        or milestone.get("partial_month_tuning_allowed") is not False
        or milestone.get("parameter_change_applies_only_to_new_generation") is not True
        or milestone.get("target_multiple_may_backsolve_sizing") is not False
    ):
        raise DojoHistoricalTrainControlError("run-control fixed boundary drifted")
    for key in ("registry_path", "source_manifest_path"):
        relative = fixed.get(key)
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise DojoHistoricalTrainControlError(f"run-control {key} is unsafe")
        if not (repo_root / relative).is_file():
            raise DojoHistoricalTrainControlError(f"run-control {key} is missing")
    output_root = execution.get("output_root")
    if not isinstance(output_root, str) or not Path(output_root).is_absolute():
        raise DojoHistoricalTrainControlError("run-control output root is invalid")
    minimum_free = execution.get("minimum_free_disk_bytes")
    if isinstance(minimum_free, bool) or not isinstance(minimum_free, int) or minimum_free < 20 * 1024**3:
        raise DojoHistoricalTrainControlError("run-control disk floor is too low")
    return control


def _registry_artifact(registry: Mapping[str, Any]) -> str:
    artifact = registry.get("artifact_sha256")
    if not isinstance(artifact, str) or len(artifact) != 64:
        raise DojoHistoricalTrainControlError("G2 registry artifact SHA is invalid")
    body = {key: value for key, value in registry.items() if key != "artifact_sha256"}
    if canonical_sha256(body) != artifact:
        raise DojoHistoricalTrainControlError("G2 registry artifact seal drifted")
    return artifact


def _candidate_proposals(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    workers = registry.get("workers")
    if not isinstance(workers, list) or len(workers) < 2:
        raise DojoHistoricalTrainControlError("G2 registry has no strategy cohort")
    proposals = []
    for worker in workers:
        if not isinstance(worker, Mapping):
            raise DojoHistoricalTrainControlError("G2 worker row is malformed")
        proposal = seal_candidate_proposal(
            {
                "contract": PROPOSAL_CONTRACT,
                "schema_version": 1,
                "candidate_id": worker.get("worker_id"),
                "family": worker.get("family"),
                "hypothesis": (
                    f"G2 fixed {worker.get('family')} family under the sealed "
                    "2x-per-position, four-slot portfolio envelope."
                ),
                "config": worker.get("config"),
                "risk_increase": False,
            }
        )
        if (
            proposal["config_sha256"] != worker.get("config_sha256")
            or proposal["catalog_sha256"] != worker.get("catalog_sha256")
        ):
            raise DojoHistoricalTrainControlError(
                "G2 worker config/catalog binding drifted"
            )
        proposals.append(proposal)
    proposals.sort(key=lambda row: row["candidate_id"])
    return proposals


def _cost_profiles() -> dict[str, dict[str, Any]]:
    return {
        "BASE": {
            "financing_policy": "ZERO_BASE_DIAGNOSTIC",
            "jpy_quote_financing_jpy_per_unit_day": 0.0,
            "non_jpy_quote_financing_jpy_per_unit_day": 0.0,
            "slippage_pips_per_adverse_fill": 0.0,
        },
        "STRESS": {
            "financing_policy": "DECLARED_FIXED_JPY_CONSERVATIVE_PROXY",
            "jpy_quote_financing_jpy_per_unit_day": 0.008,
            "non_jpy_quote_financing_jpy_per_unit_day": (
                NON_JPY_STRESS_FINANCING_JPY_PER_UNIT_DAY
            ),
            "slippage_pips_per_adverse_fill": 0.3,
        },
    }


def _risk_envelope(registry: Mapping[str, Any]) -> dict[str, Any]:
    allocator = registry.get("allocator")
    expected = {
        "leverage": 25.0,
        "margin_closeout_fraction": allocator.get("margin_closeout_fraction"),
        "max_currency_gross_notional_fraction": allocator.get(
            "maximum_gross_leverage"
        ),
        "max_cluster_gross_notional_fraction": allocator.get(
            "maximum_gross_leverage"
        ),
        "max_lock_seconds": 14_400,
        "max_margin_utilization_fraction": allocator.get(
            "new_position_margin_admission_fraction_max"
        ),
        "max_open_and_pending_per_family": allocator.get(
            "max_concurrent_per_family"
        ),
        "max_open_and_pending_per_pair": allocator.get("max_concurrent_per_pair"),
        "max_open_and_pending_total": allocator.get("simultaneous_slots"),
        "max_portfolio_stop_risk_fraction": allocator.get(
            "portfolio_stop_risk_fraction"
        ),
        "maximum_gross_leverage": allocator.get("maximum_gross_leverage"),
        "per_position_leverage": allocator.get("per_position_leverage"),
    }
    if expected != {
        "leverage": 25.0,
        "margin_closeout_fraction": 0.9,
        "max_currency_gross_notional_fraction": 8.0,
        "max_cluster_gross_notional_fraction": 8.0,
        "max_lock_seconds": 14_400,
        "max_margin_utilization_fraction": 0.45,
        "max_open_and_pending_per_family": 1,
        "max_open_and_pending_per_pair": 1,
        "max_open_and_pending_total": 4,
        "max_portfolio_stop_risk_fraction": 0.1,
        "maximum_gross_leverage": 8.0,
        "per_position_leverage": 2.0,
    }:
        raise DojoHistoricalTrainControlError("G2 allocator envelope drifted")
    return expected


def _implementation_digests(
    *,
    repo_root: Path,
    runtime_seal: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, str]:
    costs = _cost_profiles()
    values = {
        "base_cost_policy_sha256": canonical_sha256(costs["BASE"]),
        "m1_precision_policy_sha256": canonical_sha256(
            {
                "contexts": ["M1_CORE5_2020_2026H1", "M1_FULL28_2025_2026H1"],
                "context_overlap_is_not_replication": True,
                "promotion_from_m5_is_automatic": False,
            }
        ),
        "replay_engine_sha256": _file_binding(
            repo_root,
            (
                "src/quant_rabbit/dojo_long_horizon_economic_runner.py",
                "src/quant_rabbit/dojo_long_horizon_execution.py",
                "src/quant_rabbit/dojo_long_horizon_plan.py",
                "src/quant_rabbit/dojo_long_horizon_schedule.py",
                "src/quant_rabbit/dojo_market_calendar.py",
                "src/quant_rabbit/dojo_portfolio_replay_reducer.py",
                "src/quant_rabbit/dojo_shared_worker_protocol.py",
                "src/quant_rabbit/dojo_sparse_replay.py",
                "src/quant_rabbit/dojo_sparse_source_slice_v2.py",
            ),
        ),
        "risk_policy_sha256": canonical_sha256(_risk_envelope(registry)),
        "scorer_sha256": _file_binding(
            repo_root, ("src/quant_rabbit/dojo_long_horizon_result_reducer.py",)
        ),
        "strategy_bundle_sha256": runtime_seal["runtime_binding_sha256"],
        "stress_cost_policy_sha256": canonical_sha256(costs["STRESS"]),
        "trainer_sha256": _file_binding(
            repo_root,
            (
                "src/quant_rabbit/dojo_ai_trainer_packet.py",
                "src/quant_rabbit/dojo_ai_tuning_state.py",
                "src/quant_rabbit/dojo_immutable_generation_loop.py",
            ),
        ),
    }
    if set(values) != set(IMPLEMENTATION_DIGEST_KEYS):
        raise DojoHistoricalTrainControlError("implementation digest grid drifted")
    return values


def _resource_policy(control: Mapping[str, Any]) -> dict[str, int]:
    return {
        "max_resident_coordinates": 256,
        "max_rss_bytes": 8 * 1024**3,
        "max_open_files": 1024,
        "min_free_disk_bytes": control["execution"]["minimum_free_disk_bytes"],
        "max_checkpoint_bytes": 16 * 1024 * 1024,
        "max_terminal_bytes": 16 * 1024 * 1024,
        "max_parallel_jobs": 1,
    }


def prepare_generation(
    *,
    repo_root: Path,
    run_control_path: Path,
) -> dict[str, Any]:
    """Seal the G2 denominator and initialize its crash-safe execution state."""

    repo = Path(repo_root).resolve(strict=True)
    control = _verified_control(run_control_path, repo_root=repo)
    fixed = control["fixed_inputs"]
    registry = _read_json(repo / fixed["registry_path"], field="G2 registry")
    registry_sha = _registry_artifact(registry)
    if registry_sha != fixed["registry_artifact_sha256"]:
        raise DojoHistoricalTrainControlError("run control names another G2 registry")
    source_manifest = verify_long_horizon_source_manifest_seal(
        _read_json(repo / fixed["source_manifest_path"], field="source manifest")
    )
    if source_manifest["source_manifest_sha256"] != fixed["source_manifest_sha256"]:
        raise DojoHistoricalTrainControlError(
            "run control names another source manifest"
        )
    proposals = _candidate_proposals(registry)
    runtime_seal = build_tuned_strategy_runtime_seal(
        repo,
        candidate_proposals=proposals,
        generation_ordinal=GENERATION_ORDINAL,
        generation_binding_sha256=registry_sha,
    )
    runtime_seal = verify_tuned_strategy_runtime_seal(runtime_seal, repo_root=repo)
    source_inputs = long_horizon_plan_digest_inputs(source_manifest)
    families = sorted({row["family_id"] for row in runtime_seal["worker_catalog"]})
    plan = build_long_horizon_train_plan(
        portfolio_families=families,
        source_digests=source_inputs["source_digests"],
        corpus_digests=source_inputs["corpus_digests"],
        implementation_digests=_implementation_digests(
            repo_root=repo, runtime_seal=runtime_seal, registry=registry
        ),
        study_profile=fixed["study_profile"],
    )
    worker_bindings = [
        {
            key: row[key]
            for key in ("worker_id", "family_id", "config_sha256")
        }
        for row in runtime_seal["worker_catalog"]
    ]
    schedule = build_long_horizon_stream_schedule(
        plan, worker_bindings=worker_bindings
    )
    output_root = Path(control["execution"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {key: output_root / name for key, name in _ARTIFACT_NAMES.items()}
    resource_policy = _resource_policy(control)
    proposal_wrapper = {
        "contract": "QR_DOJO_G2_TRAINER_CANDIDATE_PROPOSALS_V1",
        "schema_version": 1,
        "generation": "G2",
        "registry_artifact_sha256": registry_sha,
        "trainer_candidate_proposals": proposals,
        **_AUTHORITY,
    }
    catalog_wrapper = {
        "contract": "QR_DOJO_LONG_HORIZON_WORKER_CATALOG_V1",
        "schema_version": 1,
        "worker_catalog": runtime_seal["worker_catalog"],
        **_AUTHORITY,
    }
    _write_once(paths["proposals"], proposal_wrapper)
    _write_once(paths["runtime_seal"], runtime_seal)
    _write_once(paths["worker_catalog"], catalog_wrapper)
    _write_once(paths["plan"], plan)
    _write_once(paths["schedule"], schedule)
    _write_once(paths["resource_policy"], resource_policy)
    runner_binding = {
        "runner_contract": "QR_DOJO_LONG_HORIZON_ECONOMIC_RUNNER_OUTPUT_REQUIREMENTS_V1",
        "runner_code_sha256": plan["implementation_binding"]["digests"][
            "replay_engine_sha256"
        ],
        "result_contract": CELL_CONTRACT,
    }
    initialize_long_horizon_execution_state(
        output_root / "execution-state",
        schedule=schedule,
        plan=plan,
        runner_binding=runner_binding,
        resource_policy=resource_policy,
    )
    body = {
        "contract": MANIFEST_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "classification": "WORN_HISTORICAL_TRAIN_ONLY",
        "rapid_train_evaluation_mode": "INDEPENDENT_MONTH",
        "continuous_account_role": (
            "SEPARATE_LONG_PROFILE_NOT_RAPID_TUNING_GATE"
        ),
        "non_overlapping_six_month_blocks_required": True,
        "generation": "G2",
        "registry_artifact_sha256": registry_sha,
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "run_control_sha256": hashlib.sha256(Path(run_control_path).read_bytes()).hexdigest(),
        "plan_sha256": plan["plan_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "runtime_binding_sha256": runtime_seal["runtime_binding_sha256"],
        "worker_count": len(runtime_seal["worker_catalog"]),
        "family_count": len(families),
        "stream_job_count": schedule["stream_job_count"],
        "result_coordinate_count": schedule["result_coordinate_count"],
        "trainer_milestone_policy": dict(control["trainer_milestones"]),
        "artifact_sha256": {
            key: hashlib.sha256(paths[key].read_bytes()).hexdigest()
            for key in (
                "plan",
                "proposals",
                "resource_policy",
                "runtime_seal",
                "schedule",
                "worker_catalog",
            )
        },
        **_AUTHORITY,
    }
    manifest = {**body, "manifest_sha256": canonical_sha256(body)}
    _write_once(paths["control_manifest"], manifest)
    return {
        "status": "PREPARED",
        "output_root": str(output_root),
        "plan_sha256": plan["plan_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "worker_count": len(runtime_seal["worker_catalog"]),
        "family_count": len(families),
        "stream_job_count": schedule["stream_job_count"],
        "result_coordinate_count": schedule["result_coordinate_count"],
        **_AUTHORITY,
    }


def _load_generation(
    *, repo_root: Path, run_control_path: Path
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    repo = Path(repo_root).resolve(strict=True)
    control = _verified_control(run_control_path, repo_root=repo)
    root = Path(control["execution"]["output_root"]).resolve(strict=True)
    manifest = _read_json(root / _ARTIFACT_NAMES["control_manifest"], field="control manifest")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest.get("contract") != MANIFEST_CONTRACT
        or manifest.get("manifest_sha256") != canonical_sha256(body)
    ):
        raise DojoHistoricalTrainControlError("generation control manifest drifted")
    if manifest.get("run_control_sha256") != hashlib.sha256(
        Path(run_control_path).read_bytes()
    ).hexdigest():
        raise DojoHistoricalTrainControlError(
            "generation run-control bytes differ from the prepared manifest"
        )
    artifacts = {}
    for key in (
        "plan",
        "proposals",
        "resource_policy",
        "runtime_seal",
        "schedule",
        "worker_catalog",
    ):
        path = root / _ARTIFACT_NAMES[key]
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["artifact_sha256"][key]:
            raise DojoHistoricalTrainControlError(f"generation {key} bytes drifted")
        artifacts[key] = _read_json(path, field=key)
    plan = validate_long_horizon_train_plan(artifacts["plan"])
    schedule = validate_long_horizon_stream_schedule(
        artifacts["schedule"], plan=plan
    )
    runtime_seal = verify_tuned_strategy_runtime_seal(
        artifacts["runtime_seal"], repo_root=repo
    )
    registry = _read_json(
        repo / control["fixed_inputs"]["registry_path"], field="G2 registry"
    )
    if _registry_artifact(registry) != manifest.get("registry_artifact_sha256"):
        raise DojoHistoricalTrainControlError("generation registry binding drifted")
    current_digests = _implementation_digests(
        repo_root=repo, runtime_seal=runtime_seal, registry=registry
    )
    if plan["implementation_binding"]["digests"] != current_digests:
        raise DojoHistoricalTrainControlError(
            "generation implementation bytes differ from the prepared plan"
        )
    source_manifest = verify_long_horizon_source_manifest_seal(
        _read_json(
            repo / control["fixed_inputs"]["source_manifest_path"],
            field="source manifest",
        )
    )
    return (
        control,
        root,
        plan,
        schedule,
        runtime_seal,
        source_manifest,
        artifacts["worker_catalog"],
    )


def _pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


def _conversion_routes(pairs: Sequence[str]) -> list[dict[str, str]]:
    routes = []
    for currency in sorted(
        {pair.split("_")[1] for pair in pairs if pair.split("_")[1] != "JPY"}
    ):
        direct = f"{currency}_JPY"
        inverse = f"JPY_{currency}"
        if direct in pairs:
            route_pair, orientation = direct, "JPY_PER_CURRENCY"
        elif inverse in pairs:
            route_pair, orientation = inverse, "CURRENCY_PER_JPY"
        else:
            raise DojoHistoricalTrainControlError(
                f"feed has no JPY conversion route for {currency}"
            )
        routes.append(
            {"currency": currency, "pair": route_pair, "orientation": orientation}
        )
    return routes


def _coordinate_runtimes(
    *,
    handoff: Mapping[str, Any],
    plan: Mapping[str, Any],
    catalog: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    job = handoff["job"]
    digests = plan["implementation_binding"]["digests"]
    risk = _risk_envelope(registry)
    costs = _cost_profiles()
    result = {}
    for coordinate in job["coordinates"]:
        coordinate_id = coordinate["coordinate_id"]
        if coordinate_id not in handoff["runnable_coordinate_ids"]:
            continue
        active = [
            dict(row)
            for row, bit in zip(catalog, coordinate["active_worker_mask"], strict=True)
            if bit == "1"
        ]
        trade_pairs = [
            pair
            for pair, bit in zip(job["feed_pairs"], coordinate["trade_pair_mask"], strict=True)
            if bit == "1"
        ]
        scenario = coordinate["cost_scenario"]
        cost = costs[scenario]
        slippage_pips = cost["slippage_pips_per_adverse_fill"]
        policy = seal_portfolio_policy(
            {
                "policy_id": f"g2-{scenario.lower()}-{coordinate_id[:24]}",
                "expected_quote_pairs": list(job["feed_pairs"]),
                "tradable_pairs": trade_pairs,
                "active_worker_bindings": active,
                "leverage": risk["leverage"],
                "margin_closeout_fraction": risk["margin_closeout_fraction"],
                "max_margin_utilization_fraction": risk[
                    "max_margin_utilization_fraction"
                ],
                "max_portfolio_stop_risk_fraction": risk[
                    "max_portfolio_stop_risk_fraction"
                ],
                "max_open_and_pending_total": risk["max_open_and_pending_total"],
                "max_open_and_pending_per_pair": risk[
                    "max_open_and_pending_per_pair"
                ],
                "max_open_and_pending_per_family": risk[
                    "max_open_and_pending_per_family"
                ],
                "max_currency_gross_notional_fraction": risk[
                    "max_currency_gross_notional_fraction"
                ],
                "max_cluster_gross_notional_fraction": risk[
                    "max_cluster_gross_notional_fraction"
                ],
                "max_lock_seconds": risk["max_lock_seconds"],
                "slippage_by_pair": [
                    {
                        "pair": pair,
                        "entry_slippage_price": slippage_pips * _pip_size(pair),
                        "exit_slippage_price": slippage_pips * _pip_size(pair),
                    }
                    for pair in job["feed_pairs"]
                ],
                "financing_by_pair": [
                    {
                        "pair": pair,
                        "long_cost_jpy_per_unit_day": (
                            cost["jpy_quote_financing_jpy_per_unit_day"]
                            if pair.endswith("_JPY")
                            else cost["non_jpy_quote_financing_jpy_per_unit_day"]
                        ),
                        "short_cost_jpy_per_unit_day": (
                            cost["jpy_quote_financing_jpy_per_unit_day"]
                            if pair.endswith("_JPY")
                            else cost["non_jpy_quote_financing_jpy_per_unit_day"]
                        ),
                    }
                    for pair in job["feed_pairs"]
                ],
                "conversion_routes": _conversion_routes(job["feed_pairs"]),
                "correlation_bindings": [],
            }
        )
        cost_sha = digests[
            "base_cost_policy_sha256"
            if scenario == "BASE"
            else "stress_cost_policy_sha256"
        ]
        binding_body = {
            "coordinate_id": coordinate_id,
            "cost_scenario": scenario,
            "portfolio_policy_sha256": policy["policy_sha256"],
            "cost_policy_sha256": cost_sha,
            "risk_policy_sha256": digests["risk_policy_sha256"],
            "replay_engine_sha256": digests["replay_engine_sha256"],
            "allocator_policy": coordinate["allocator_policy"],
            "initial_balance_jpy": 200_000,
        }
        result[coordinate_id] = {
            "coordinate_id": coordinate_id,
            "cost_scenario": scenario,
            "trade_pairs": trade_pairs,
            "portfolio_policy": policy,
            "cost_policy_sha256": cost_sha,
            "risk_policy_sha256": digests["risk_policy_sha256"],
            "replay_engine_sha256": digests["replay_engine_sha256"],
            "portfolio_policy_binding_sha256": canonical_sha256(binding_body),
        }
    return result


def _verified_job_result(
    value: Mapping[str, Any], *, handoff: Mapping[str, Any]
) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "economic_job_result_sha256"}
    if (
        value.get("contract") != ECONOMIC_JOB_RESULT_CONTRACT
        or value.get("economic_job_result_sha256") != canonical_sha256(body)
        or value.get("job_sha256") != handoff["job"]["job_sha256"]
        or value.get("claim_sha256") != handoff["claim"]["claim_sha256"]
        or value.get("coordinate_result_count")
        != len(handoff["runnable_coordinate_ids"])
    ):
        raise DojoHistoricalTrainControlError("economic job result is invalid")
    return dict(value)


def _carry_inputs(root: Path, handoff: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    slots = sorted(
        {
            row["predecessor_state_slot_id"]
            for row in handoff["job"]["coordinates"]
            if row["coordinate_id"] in handoff["runnable_coordinate_ids"]
            and row["predecessor_state_slot_id"] is not None
        }
    )
    result = {}
    for slot in slots:
        result[slot] = _read_json(
            root / "economic-carry-states" / f"{slot}.json",
            field=f"economic carry {slot}",
        )
    return result


def _publish_carries(root: Path, result: Mapping[str, Any]) -> None:
    carries = result.get("economic_carry_states_by_slot")
    if not isinstance(carries, Mapping):
        raise DojoHistoricalTrainControlError("economic result carry map is invalid")
    for slot, carry in carries.items():
        if not isinstance(slot, str) or not isinstance(carry, Mapping):
            raise DojoHistoricalTrainControlError("economic carry row is invalid")
        _write_once(root / "economic-carry-states" / f"{slot}.json", carry)


def _milestone_status(
    root: Path, schedule: Mapping[str, Any], *, publish: bool = True
) -> dict[str, Any]:
    jobs_by_sha = {job["job_sha256"]: job for job in schedule["jobs"]}
    completed_paths: dict[str, set[str]] = {}
    completed_jobs = 0
    result_root = root / "job-results"
    if result_root.is_dir():
        for path in sorted(result_root.glob("*.json")):
            value = _read_json(path, field="job result")
            job = jobs_by_sha.get(value.get("job_sha256"))
            if job is None or value.get("job_status") != "COMPLETE":
                continue
            completed_jobs += 1
            if job["source_binding_id"] == "M5_EXACT28_2020_2026H1":
                completed_paths.setdefault(job["month"], set()).add(
                    job["intrabar_path"]
                )
    completed_months = sorted(
        month for month, paths in completed_paths.items() if paths == {"OHLC", "OLHC"}
    )
    block_count = len(completed_months) // 6
    body = {
        "contract": MILESTONE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "classification": "WORN_HISTORICAL_TRAIN_ONLY",
        "rapid_train_evaluation_mode": "INDEPENDENT_MONTH",
        "continuous_account_role": (
            "SEPARATE_LONG_PROFILE_NOT_RAPID_TUNING_GATE"
        ),
        "non_overlapping_six_month_blocks_required": True,
        "completed_job_count": completed_jobs,
        "completed_m5_month_count": len(completed_months),
        "completed_m5_months": completed_months,
        "complete_six_month_trainer_block_count": block_count,
        "next_trainer_review_at_completed_m5_month_count": (block_count + 1) * 6,
        "trainer_review_due": len(completed_months) > 0 and len(completed_months) % 6 == 0,
        "partial_month_economics_may_tune": False,
        "parameter_change_requires_new_generation": True,
        "three_x_status": "3X_NOT_REACHABLE",
        **_AUTHORITY,
    }
    milestone = {**body, "milestone_sha256": canonical_sha256(body)}
    if publish:
        _write_once(
            root
            / "trainer-milestones"
            / f"jobs-{completed_jobs:04d}-{milestone['milestone_sha256']}.json",
            milestone,
        )
    return milestone


def _seal_source_failure(
    *,
    root: Path,
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
    handoff: Mapping[str, Any],
    error: Exception,
) -> dict[str, Any]:
    """Count a deterministic source failure in the fixed denominator."""

    job = handoff["job"]
    claim = handoff["claim"]
    runnable_count = len(handoff["runnable_coordinate_ids"])
    coordinate_count = len(job["coordinates"])
    recorded_count = handoff["recorded_coordinate_count"]
    if (
        handoff["pending_coordinate_count"] != runnable_count
        or handoff["predecessor_blocked_coordinate_count"] != 0
        or recorded_count + runnable_count != coordinate_count
    ):
        raise DojoHistoricalTrainControlError(
            "source failure denominator is neither recorded-predecessor plus runnable"
        )
    message = str(error)
    if not message or len(message) > 4096:
        raise DojoHistoricalTrainControlError(
            "source failure message is empty or exceeds its evidence bound"
        )
    evidence_body = {
        "contract": "QR_DOJO_HISTORICAL_SOURCE_FAILURE_V1",
        "schema_version": SCHEMA_VERSION,
        "job_sha256": job["job_sha256"],
        "claim_sha256": claim["claim_sha256"],
        "stage": "SPARSE_SOURCE_MATERIALIZATION",
        "error_type": type(error).__name__,
        "error": message,
        **_AUTHORITY,
    }
    evidence = {
        **evidence_body,
        "failure_evidence_sha256": canonical_sha256(evidence_body),
    }
    job_dir = root / "jobs" / job["job_sha256"]
    _write_once(job_dir / "source-failure.json", evidence)
    cells = [
        build_long_horizon_coordinate_result(
            job=job,
            claim=claim,
            coordinate_id=coordinate_id,
            status="FAILED",
            failure={
                "code": "SOURCE_SLICE_MATERIALIZATION_FAILURE",
                "retryable": False,
                "evidence_sha256": evidence["failure_evidence_sha256"],
            },
        )
        for coordinate_id in handoff["runnable_coordinate_ids"]
    ]
    record_long_horizon_coordinate_results(
        root / "execution-state",
        schedule=schedule,
        plan=plan,
        claim_sha256=claim["claim_sha256"],
        results=cells,
    )
    terminal = seal_long_horizon_attempt(
        root / "execution-state",
        schedule=schedule,
        plan=plan,
        claim_sha256=claim["claim_sha256"],
    )
    terminal_cells = terminal["terminal_manifest"]["cells"]
    complete_coordinate_count = sum(
        row["status"] == "COMPLETE" for row in terminal_cells
    )
    failed_coordinate_count = sum(row["status"] == "FAILED" for row in terminal_cells)
    completion_body = {
        "contract": JOB_COMPLETION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "job_sha256": job["job_sha256"],
        "claim_sha256": claim["claim_sha256"],
        "source_binding_id": job["source_binding_id"],
        "month": job["month"],
        "intrabar_path": job["intrabar_path"],
        "coordinate_result_count": len(terminal_cells),
        "new_source_failure_coordinate_count": len(cells),
        "predecessor_failure_coordinate_count": recorded_count,
        "complete_coordinate_count": complete_coordinate_count,
        "failed_coordinate_count": failed_coordinate_count,
        "job_status": "FAILED_SOURCE",
        "economic_job_result_sha256": None,
        "terminal_sha256": terminal["terminal_manifest"]["terminal_sha256"],
        "free_disk_bytes_after": shutil.disk_usage(root).free,
        **_AUTHORITY,
    }
    completion = {
        **completion_body,
        "completion_sha256": canonical_sha256(completion_body),
    }
    _write_once(job_dir / "completion.json", completion)
    milestone = _milestone_status(root, schedule)
    return {
        "status": "JOB_FAILED_SOURCE",
        "output_root": str(root),
        "job": {
            "job_sha256": job["job_sha256"],
            "source_binding_id": job["source_binding_id"],
            "month": job["month"],
            "intrabar_path": job["intrabar_path"],
        },
        "coordinate_result_count": len(terminal_cells),
        "new_source_failure_coordinate_count": len(cells),
        "predecessor_failure_coordinate_count": recorded_count,
        "complete_coordinate_count": complete_coordinate_count,
        "failed_coordinate_count": failed_coordinate_count,
        "failure_evidence_sha256": evidence["failure_evidence_sha256"],
        "trainer_milestone": milestone,
        **_AUTHORITY,
    }


def run_next_job(
    *,
    repo_root: Path,
    run_control_path: Path,
) -> dict[str, Any]:
    """Execute exactly one claimed historical job under the sealed G2 policy."""

    (
        control,
        root,
        plan,
        schedule,
        runtime_seal,
        source_manifest,
        catalog_wrapper,
    ) = _load_generation(repo_root=repo_root, run_control_path=run_control_path)
    lock_path = root / ".historical-train.lock"
    lock_descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DojoHistoricalTrainControlError(
                "another historical TRAIN job already owns the run lock"
            ) from exc
        free_bytes = shutil.disk_usage(root).free
        disk_floor = control["execution"]["minimum_free_disk_bytes"]
        if free_bytes < disk_floor:
            raise DojoHistoricalTrainControlError(
                f"free disk {free_bytes} is below the sealed floor {disk_floor}"
            )
        handoff = claim_next_long_horizon_job(
            root / "execution-state",
            schedule=schedule,
            plan=plan,
            runner_id=RUNNER_ID,
        )
        job = handoff["job"]
        claim = handoff["claim"]
        job_sha = job["job_sha256"]
        job_dir = root / "jobs" / job_sha
        job_dir.mkdir(parents=True, exist_ok=True)
        _write_once(job_dir / "runner-handoff.json", handoff)
        source_root = root / "source-slices"
        relative = Path(
            job["source_binding_id"],
            job["month"],
            f"{job['intrabar_path']}-{job_sha}.jsonl",
        )
        (source_root / relative.parent).mkdir(parents=True, exist_ok=True)
        receipt_path = job_dir / "source-slice-receipt.json"
        if receipt_path.exists():
            receipt = validate_sparse_month_source_slice_receipt(
                _read_json(receipt_path, field="source slice receipt"),
                job=job,
                source_manifest=source_manifest,
            )
        else:
            if (source_root / relative).exists():
                raise DojoHistoricalTrainControlError(
                    "orphan source slice exists without its immutable receipt"
                )
            try:
                receipt = build_sparse_month_source_slice_receipt(
                    source_root=source_root,
                    relative_path=relative.as_posix(),
                    job=job,
                    source_manifest=source_manifest,
                )
            except (DojoLongHorizonEconomicRunnerError, SparseReplayError) as exc:
                return _seal_source_failure(
                    root=root,
                    schedule=schedule,
                    plan=plan,
                    handoff=handoff,
                    error=exc,
                )
            _write_once(receipt_path, receipt)
        repo = Path(repo_root).resolve(strict=True)
        registry = _read_json(
            repo / control["fixed_inputs"]["registry_path"], field="G2 registry"
        )
        catalog = catalog_wrapper.get("worker_catalog")
        if not isinstance(catalog, list):
            raise DojoHistoricalTrainControlError("worker catalog is invalid")
        runtimes = _coordinate_runtimes(
            handoff=handoff, plan=plan, catalog=catalog, registry=registry
        )
        _write_once(
            job_dir / "coordinate-runtimes.json",
            {"coordinate_runtimes": runtimes},
        )
        carries = _carry_inputs(root, handoff)
        if carries:
            _write_once(
                job_dir / "carry-states-input.json",
                {"economic_carry_states_by_slot": carries},
            )
        evidence_root = job_dir / "economic-evidence"
        evidence_root.mkdir(parents=True, exist_ok=True)
        result_path = root / "job-results" / f"{job_sha}.json"
        if result_path.exists():
            result = _verified_job_result(
                _read_json(result_path, field="economic job result"), handoff=handoff
            )
        else:
            runtime_factory = build_tuned_strategy_runtime_factory(
                runtime_seal, repo_root=repo
            )
            result = run_long_horizon_economic_job(
                runner_handoff=handoff,
                plan=plan,
                source_root=source_root,
                source_manifest=source_manifest,
                source_slice_receipt=receipt,
                economic_evidence_root=evidence_root,
                worker_catalog=catalog,
                coordinate_runtimes=runtimes,
                worker_runtime_factory=runtime_factory,
                worker_runtime_binding_sha256=runtime_seal[
                    "runtime_binding_sha256"
                ],
                worker_runtime_seal=runtime_seal,
                worker_runtime_repo_root=repo,
                carry_states_by_slot=carries,
            )
            result = _verified_job_result(result, handoff=handoff)
            _write_once(result_path, result)
        _publish_carries(root, result)
        record_long_horizon_coordinate_results(
            root / "execution-state",
            schedule=schedule,
            plan=plan,
            claim_sha256=claim["claim_sha256"],
            results=result["coordinate_results"],
        )
        terminal = seal_long_horizon_attempt(
            root / "execution-state",
            schedule=schedule,
            plan=plan,
            claim_sha256=claim["claim_sha256"],
        )
        completion_body = {
            "contract": JOB_COMPLETION_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "job_sha256": job_sha,
            "claim_sha256": claim["claim_sha256"],
            "source_binding_id": job["source_binding_id"],
            "month": job["month"],
            "intrabar_path": job["intrabar_path"],
            "coordinate_result_count": result["coordinate_result_count"],
            "complete_coordinate_count": result["complete_coordinate_count"],
            "failed_coordinate_count": result["failed_coordinate_count"],
            "job_status": result["job_status"],
            "economic_job_result_sha256": result["economic_job_result_sha256"],
            "terminal_sha256": terminal["terminal_manifest"]["terminal_sha256"],
            "free_disk_bytes_after": shutil.disk_usage(root).free,
            **_AUTHORITY,
        }
        completion = {
            **completion_body,
            "completion_sha256": canonical_sha256(completion_body),
        }
        _write_once(job_dir / "completion.json", completion)
        milestone = _milestone_status(root, schedule)
        return {
            "status": "JOB_COMPLETE",
            "output_root": str(root),
            "job": {
                "job_sha256": job_sha,
                "source_binding_id": job["source_binding_id"],
                "month": job["month"],
                "intrabar_path": job["intrabar_path"],
            },
            "coordinate_result_count": result["coordinate_result_count"],
            "complete_coordinate_count": result["complete_coordinate_count"],
            "failed_coordinate_count": result["failed_coordinate_count"],
            "trainer_milestone": milestone,
            **_AUTHORITY,
        }
    finally:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock_descriptor)


def generation_status(
    *, repo_root: Path, run_control_path: Path
) -> dict[str, Any]:
    """Return verified compact progress without opening partial economics."""

    _, root, plan, schedule, _, _, _ = _load_generation(
        repo_root=repo_root, run_control_path=run_control_path
    )
    execution = long_horizon_execution_status(
        root / "execution-state", schedule=schedule, plan=plan
    )
    milestone = _milestone_status(root, schedule, publish=False)
    return {
        "contract": "QR_DOJO_HISTORICAL_TRAIN_STATUS_V1",
        "schema_version": SCHEMA_VERSION,
        "output_root": str(root),
        "execution": execution,
        "trainer_milestone": milestone,
        "free_disk_bytes": shutil.disk_usage(root).free,
        **_AUTHORITY,
    }


__all__ = [
    "DojoHistoricalTrainControlError",
    "generation_status",
    "prepare_generation",
    "run_next_job",
]
