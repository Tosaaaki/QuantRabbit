"""Fixed-denominator portfolio counterfactual for room-meta-01 arms.

A held candidate cannot be valued in isolation because HOLD changes free margin,
later admissions, and the state of the whole account.  This scorer therefore
accepts two independently reexecuted economic job results over the same sealed
source and coordinate denominator: ``BASE_BOT`` and one anomaly admission arm.
It validates both complete portfolios and reports paired coordinate deltas.

The result is research evidence only.  Coordinate-grid deltas are never summed
into a fictitious profit claim and this module grants no promotion or live
authority.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any, Final

from quant_rabbit.dojo_anomaly_admission_controller import EXPERIMENT_ARMS
from quant_rabbit.dojo_anomaly_admission_runtime import EVIDENCE_SUMMARY_CONTRACT
from quant_rabbit.dojo_anomaly_decision_scorer import FIXED_ATTESTATION_CONTRACT
from quant_rabbit.dojo_long_horizon_economic_runner import (
    ECONOMIC_JOB_RESULT_CONTRACT,
)
from quant_rabbit.dojo_portfolio_replay_reducer import (
    canonical_portfolio_sha256,
    validate_portfolio_replay_result,
)


COUNTERFACTUAL_CONTRACT: Final = (
    "QR_DOJO_ANOMALY_ARM_PORTFOLIO_COUNTERFACTUAL_V1"
)
SCHEMA_VERSION: Final = 1
RUNTIME_MODE: Final = "SEALED_ANOMALY_ADMISSION_OVER_TUNED_STRATEGY"
MAX_RESULT_FILE_BYTES: Final = 16 * 1024 * 1024
_PAIR_BINDING_FIELDS: Final = (
    "runner_handoff_sha256",
    "plan_sha256",
    "implementation_binding_sha256",
    "job_sha256",
    "claim_sha256",
    "source_slice_receipt_sha256",
    "source_row_count",
    "quote_batch_count",
    "batch_chain_sha256",
    "coordinate_runtime_bindings_sha256",
    "worker_upstream_runtime_binding_sha256",
    "worker_runtime_capacity_slots",
    "worker_runtime_policy_sha256",
    "economic_transcript_format",
    "sparse_observed_epoch_union_used",
    "synthetic_executable_quote_count",
    "carry_forward_executable_quote_count",
)
_AUTHORITY: Final = {
    "research_only": True,
    "historical_train_is_forward_proof": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
    "automatic_deployment_allowed": False,
}


class DojoAnomalyPortfolioCounterfactualError(ValueError):
    """The paired jobs are incomplete, mutable, or not causally comparable."""


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DojoAnomalyPortfolioCounterfactualError(f"{field} must be an object")
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise DojoAnomalyPortfolioCounterfactualError(
            f"{field} is not strict canonical JSON"
        ) from exc


def _sequence(value: Any, *, field: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DojoAnomalyPortfolioCounterfactualError(f"{field} must be an array")
    return list(value)


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DojoAnomalyPortfolioCounterfactualError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DojoAnomalyPortfolioCounterfactualError(f"{field} must be finite")
    return result


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DojoAnomalyPortfolioCounterfactualError(
            f"{field} must be a non-negative integer"
        )
    return value


def _verified_evidence(
    value: Any,
    *,
    coordinate_id: str,
    arm: str,
    runtime_binding_sha256: str,
) -> dict[str, Any]:
    evidence = _mapping(value, field=f"runtime evidence {coordinate_id}")
    claimed = evidence.get("evidence_summary_sha256")
    body = {
        key: item
        for key, item in evidence.items()
        if key != "evidence_summary_sha256"
    }
    if (
        evidence.get("contract") != EVIDENCE_SUMMARY_CONTRACT
        or claimed != canonical_portfolio_sha256(body)
        or evidence.get("arm") != arm
        or evidence.get("runtime_binding_sha256") != runtime_binding_sha256
        or evidence.get("runner_integration_complete") is not True
        or evidence.get("official_evidence_eligible") is not False
    ):
        raise DojoAnomalyPortfolioCounterfactualError(
            f"runtime evidence is invalid for {coordinate_id}"
        )
    counts = _mapping(
        evidence.get("counts"), field=f"runtime evidence counts {coordinate_id}"
    )
    for key in ("decisions", "upstream_candidates", "selected", "held", "reduced"):
        _integer(counts.get(key), field=f"runtime evidence {coordinate_id}.{key}")
    return evidence


def _verified_job_result(value: Any, *, label: str) -> dict[str, Any]:
    result = _mapping(value, field=f"{label} economic job result")
    claimed = result.get("economic_job_result_sha256")
    body = {
        key: item
        for key, item in result.items()
        if key != "economic_job_result_sha256"
    }
    if (
        result.get("contract") != ECONOMIC_JOB_RESULT_CONTRACT
        or result.get("schema_version") != 1
        or claimed != canonical_portfolio_sha256(body)
        or result.get("job_status") != "COMPLETE"
        or result.get("failed_coordinate_count") != 0
        or result.get("partial_economics_reported") is not False
        or result.get("downstream_terminal_reduction_allowed") is not True
        or result.get("independent_economic_reexecution_passed") is not True
        or result.get("strategy_decision_reexecution_passed") is not True
        or result.get("source_quote_coverage_proved") is not True
        or result.get("worker_runtime_mode") != RUNTIME_MODE
        or result.get("synthetic_executable_quote_count") != 0
        or result.get("carry_forward_executable_quote_count") != 0
    ):
        raise DojoAnomalyPortfolioCounterfactualError(
            f"{label} job lacks complete independent economic/decision evidence"
        )
    arm = result.get("worker_runtime_arm")
    if arm not in EXPERIMENT_ARMS or arm == "AI_EXIT_CAPITAL_RELEASE":
        raise DojoAnomalyPortfolioCounterfactualError(
            f"{label} job has an unsupported admission arm"
        )
    runtime_sha = result.get("worker_runtime_binding_sha256")
    if not isinstance(runtime_sha, str) or len(runtime_sha) != 64:
        raise DojoAnomalyPortfolioCounterfactualError(
            f"{label} job runtime binding is invalid"
        )
    for field in (
        "worker_runtime_policy_sha256",
        "worker_upstream_runtime_binding_sha256",
    ):
        value_sha = result.get(field)
        if not isinstance(value_sha, str) or len(value_sha) != 64:
            raise DojoAnomalyPortfolioCounterfactualError(
                f"{label} job {field} is invalid"
            )
    capacity_slots = result.get("worker_runtime_capacity_slots")
    if (
        isinstance(capacity_slots, bool)
        or not isinstance(capacity_slots, int)
        or capacity_slots <= 0
    ):
        raise DojoAnomalyPortfolioCounterfactualError(
            f"{label} job capacity slot binding is invalid"
        )
    fixed = _mapping(
        result.get("fixed_denominator_decision_reexecution_attestation"),
        field=f"{label} fixed decision attestation",
    )
    fixed_claimed = fixed.get("fixed_denominator_decision_attestation_sha256")
    fixed_body = {
        key: item
        for key, item in fixed.items()
        if key != "fixed_denominator_decision_attestation_sha256"
    }
    if (
        fixed.get("contract") != FIXED_ATTESTATION_CONTRACT
        or fixed_claimed != canonical_portfolio_sha256(fixed_body)
        or fixed_claimed
        != result.get(
            "fixed_denominator_decision_reexecution_attestation_sha256"
        )
        or fixed.get("status") != "VERIFIED_COMPLETE"
        or fixed.get("fixed_denominator_decision_reexecution_passed") is not True
        or fixed.get("partial_decision_evidence_reported") is not False
    ):
        raise DojoAnomalyPortfolioCounterfactualError(
            f"{label} fixed decision attestation is invalid"
        )
    cells = _sequence(result.get("coordinate_results"), field=f"{label} cells")
    coordinate_ids = sorted(
        row.get("coordinate_id")
        for row in cells
        if isinstance(row, Mapping) and isinstance(row.get("coordinate_id"), str)
    )
    if (
        not coordinate_ids
        or len(coordinate_ids) != len(cells)
        or len(coordinate_ids) != len(set(coordinate_ids))
        or result.get("coordinate_result_count") != len(coordinate_ids)
        or result.get("complete_coordinate_count") != len(coordinate_ids)
        or fixed.get("expected_coordinate_count") != len(coordinate_ids)
        or fixed.get("verified_coordinate_count") != len(coordinate_ids)
    ):
        raise DojoAnomalyPortfolioCounterfactualError(
            f"{label} coordinate denominator is incomplete"
        )
    decision_map = _mapping(
        fixed.get("decision_reexecution_attestation_sha256_by_coordinate"),
        field=f"{label} fixed decision coordinate map",
    )
    economic_map = _mapping(
        fixed.get("economic_reexecution_attestation_sha256_by_coordinate"),
        field=f"{label} fixed economic coordinate map",
    )
    if (
        fixed.get("expected_coordinate_ids_sha256")
        != canonical_portfolio_sha256(coordinate_ids)
        or set(decision_map) != set(coordinate_ids)
        or set(economic_map) != set(coordinate_ids)
    ):
        raise DojoAnomalyPortfolioCounterfactualError(
            f"{label} fixed attestation coordinate map is incomplete"
        )
    portfolios = _mapping(
        result.get("portfolio_results_by_coordinate"),
        field=f"{label} portfolios",
    )
    evidence = _mapping(
        result.get("worker_runtime_evidence_by_coordinate"),
        field=f"{label} runtime evidence",
    )
    expected = set(coordinate_ids)
    if set(portfolios) != expected or set(evidence) != expected:
        raise DojoAnomalyPortfolioCounterfactualError(
            f"{label} portfolios/evidence do not equal the fixed denominator"
        )
    for coordinate_id in coordinate_ids:
        try:
            validate_portfolio_replay_result(portfolios[coordinate_id])
        except (TypeError, ValueError) as exc:
            raise DojoAnomalyPortfolioCounterfactualError(
                f"{label} portfolio is invalid for {coordinate_id}"
            ) from exc
        _verified_evidence(
            evidence[coordinate_id],
            coordinate_id=coordinate_id,
            arm=arm,
            runtime_binding_sha256=runtime_sha,
        )
    result["_verified_coordinate_ids"] = coordinate_ids
    return result


def _delta(candidate: Any, baseline: Any, *, field: str) -> float:
    return _finite(candidate, field=f"candidate {field}") - _finite(
        baseline, field=f"baseline {field}"
    )


def score_anomaly_arm_portfolio_counterfactual(
    *,
    baseline_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and compare two complete, independently reexecuted accounts."""

    baseline = _verified_job_result(baseline_result, label="baseline")
    candidate = _verified_job_result(candidate_result, label="candidate")
    if baseline["worker_runtime_arm"] != "BASE_BOT":
        raise DojoAnomalyPortfolioCounterfactualError(
            "baseline arm must be BASE_BOT"
        )
    if candidate["worker_runtime_arm"] == "BASE_BOT":
        raise DojoAnomalyPortfolioCounterfactualError(
            "candidate arm must change admission behavior"
        )
    if (
        baseline["worker_runtime_binding_sha256"]
        == candidate["worker_runtime_binding_sha256"]
    ):
        raise DojoAnomalyPortfolioCounterfactualError(
            "paired arms must have different sealed runtime bindings"
        )
    for field in _PAIR_BINDING_FIELDS:
        if baseline.get(field) != candidate.get(field):
            raise DojoAnomalyPortfolioCounterfactualError(
                f"paired economic jobs differ at {field}"
            )
    coordinate_ids = baseline.pop("_verified_coordinate_ids")
    if candidate.pop("_verified_coordinate_ids") != coordinate_ids:
        raise DojoAnomalyPortfolioCounterfactualError(
            "paired economic jobs have different coordinate denominators"
        )

    coordinate_rows = []
    for coordinate_id in coordinate_ids:
        base = baseline["portfolio_results_by_coordinate"][coordinate_id]
        arm = candidate["portfolio_results_by_coordinate"][coordinate_id]
        for field in (
            "policy_sha256",
            "terminal_policy",
            "processed_coordinate_count",
            "start_epoch",
            "end_epoch",
            "duration_seconds",
            "start_equity_jpy",
        ):
            if base[field] != arm[field]:
                raise DojoAnomalyPortfolioCounterfactualError(
                    f"paired portfolio differs at {coordinate_id}.{field}"
                )
        base_evidence = baseline["worker_runtime_evidence_by_coordinate"][coordinate_id]
        arm_evidence = candidate["worker_runtime_evidence_by_coordinate"][coordinate_id]
        row = {
            "coordinate_id": coordinate_id,
            "baseline_result_sha256": base["result_sha256"],
            "candidate_result_sha256": arm["result_sha256"],
            "baseline_end_equity_jpy": base["end_equity_jpy"],
            "candidate_end_equity_jpy": arm["end_equity_jpy"],
            "end_equity_delta_jpy": _delta(
                arm["end_equity_jpy"], base["end_equity_jpy"], field="end equity"
            ),
            "realized_pnl_delta_jpy": _delta(
                arm["realized_pnl_jpy"],
                base["realized_pnl_jpy"],
                field="realized pnl",
            ),
            "transaction_cost_delta_jpy": _delta(
                arm["transaction_cost_jpy"],
                base["transaction_cost_jpy"],
                field="transaction cost",
            ),
            "max_drawdown_fraction_delta": _delta(
                arm["max_drawdown_fraction"],
                base["max_drawdown_fraction"],
                field="max drawdown",
            ),
            "capital_lock_margin_jpy_hours_delta": _delta(
                arm["capital_lock_margin_jpy_hours"],
                base["capital_lock_margin_jpy_hours"],
                field="capital lock",
            ),
            "peak_margin_usage_fraction_delta": _delta(
                arm["peak_margin_usage_fraction"],
                base["peak_margin_usage_fraction"],
                field="peak margin usage",
            ),
            "trade_count_delta": arm["trade_count"] - base["trade_count"],
            "margin_reject_count_delta": (
                arm["margin_reject_count"] - base["margin_reject_count"]
            ),
            "margin_closeout_count_delta": (
                arm["margin_closeouts"] - base["margin_closeouts"]
            ),
            "ruin_event_count_delta": (
                arm["ruin_event_count"] - base["ruin_event_count"]
            ),
            "baseline_held_decision_count": base_evidence["counts"]["held"],
            "candidate_held_decision_count": arm_evidence["counts"]["held"],
            "candidate_reduced_decision_count": arm_evidence["counts"]["reduced"],
        }
        coordinate_rows.append(row)

    equity_deltas = [row["end_equity_delta_jpy"] for row in coordinate_rows]
    tolerance = 1e-9
    body = {
        "contract": COUNTERFACTUAL_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "job_sha256": baseline["job_sha256"],
        "source_slice_receipt_sha256": baseline["source_slice_receipt_sha256"],
        "batch_chain_sha256": baseline["batch_chain_sha256"],
        "baseline_arm": baseline["worker_runtime_arm"],
        "candidate_arm": candidate["worker_runtime_arm"],
        "baseline_runtime_binding_sha256": baseline[
            "worker_runtime_binding_sha256"
        ],
        "candidate_runtime_binding_sha256": candidate[
            "worker_runtime_binding_sha256"
        ],
        "upstream_runtime_binding_sha256": baseline[
            "worker_upstream_runtime_binding_sha256"
        ],
        "policy_sha256": baseline["worker_runtime_policy_sha256"],
        "capacity_slots": baseline["worker_runtime_capacity_slots"],
        "coordinate_count": len(coordinate_rows),
        "coordinate_rows": coordinate_rows,
        "candidate_improved_coordinate_count": sum(
            value > tolerance for value in equity_deltas
        ),
        "candidate_equal_coordinate_count": sum(
            abs(value) <= tolerance for value in equity_deltas
        ),
        "candidate_worse_coordinate_count": sum(
            value < -tolerance for value in equity_deltas
        ),
        "minimum_end_equity_delta_jpy": min(equity_deltas),
        "median_end_equity_delta_jpy": median(equity_deltas),
        "maximum_end_equity_delta_jpy": max(equity_deltas),
        "candidate_lower_drawdown_coordinate_count": sum(
            row["max_drawdown_fraction_delta"] < -tolerance
            for row in coordinate_rows
        ),
        "candidate_freed_capital_lock_coordinate_count": sum(
            row["capital_lock_margin_jpy_hours_delta"] < -tolerance
            for row in coordinate_rows
        ),
        "fixed_denominator_preserved": True,
        "independent_economic_reexecution_passed": True,
        "independent_decision_reexecution_passed": True,
        "held_economic_counterfactual_reexecution_passed": True,
        "partial_counterfactual_reported": False,
        "coordinate_grid_profit_sum_allowed": False,
        "three_x_claim_allowed": False,
        "monthly_three_x_proven": False,
        "official_evidence_eligible": False,
        "interpretation": (
            "PAIRED_ACCOUNT_DELTAS_ONLY_NO_CROSS_COORDINATE_PROFIT_SUM"
        ),
        "authority": dict(_AUTHORITY),
    }
    return {
        **body,
        "counterfactual_attestation_sha256": canonical_portfolio_sha256(body),
    }


def _read_result(path: Path) -> dict[str, Any]:
    target = Path(path).resolve(strict=True)
    before = target.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > MAX_RESULT_FILE_BYTES
    ):
        raise DojoAnomalyPortfolioCounterfactualError(
            "economic result file must be bounded, regular, and single-link"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags)
    try:
        payload = b""
        while chunk := os.read(descriptor, 1024 * 1024):
            payload += chunk
            if len(payload) > MAX_RESULT_FILE_BYTES:
                raise DojoAnomalyPortfolioCounterfactualError(
                    "economic result file exceeds its read bound"
                )
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = target.stat(follow_symlinks=False)
    identities = {
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        for item in (before, opened, after)
    }
    if len(identities) != 1 or len(payload) != before.st_size:
        raise DojoAnomalyPortfolioCounterfactualError(
            "economic result file changed while reading"
        )
    try:
        return _mapping(json.loads(payload), field="economic result file")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DojoAnomalyPortfolioCounterfactualError(
            "economic result file is not strict JSON"
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--candidate-result", type=Path, required=True)
    args = parser.parse_args(argv)
    result = score_anomaly_arm_portfolio_counterfactual(
        baseline_result=_read_result(args.baseline_result),
        candidate_result=_read_result(args.candidate_result),
    )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COUNTERFACTUAL_CONTRACT",
    "DojoAnomalyPortfolioCounterfactualError",
    "score_anomaly_arm_portfolio_counterfactual",
]
