#!/usr/bin/env python3
"""Lock and evaluate one untouched forward forecast candidate.

The normal replay script is deliberately exploratory.  This companion has two
separate phases: ``lock`` freezes one candidate before its forecast window, and
``evaluate`` later scores only that exact candidate with no threshold, pair,
period, or exit-policy override flags.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import oanda_history_replay_validate as replay  # noqa: E402


LOCK_KIND = "QR_FORECAST_FORWARD_HOLDOUT_V1"
RESULT_KIND = "QR_FORECAST_FORWARD_HOLDOUT_RESULT_V1"
RESULT_HEAD_KIND = "QR_FORECAST_FORWARD_HOLDOUT_HEAD_V1"
RESULT_REGISTRY_KIND = "QR_FORECAST_FORWARD_HOLDOUT_RESULT_REGISTRY_V1"
CANDIDATE_KIND = "QR_FORECAST_FORWARD_CANDIDATE_V1"
LOCK_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
SEMANTICS_VERSION = "oanda-bidask-technical-context-forward-v1"
SELECTOR_FIELDS = (
    "technical_regime",
    "technical_atr_band",
    "technical_spread_band",
    "technical_range_location_24h",
    "technical_structure_alignment",
)
CANDIDATE_KEYS = {
    "contract_kind",
    "candidate_id",
    "pair",
    "direction",
    "confidence_policy",
    "technical_selector",
    "max_horizon_min",
    "exit_policy",
    "acceptance",
}
CONFIDENCE_KEYS = {"field", "minimum"}
EXIT_KEYS = {"take_profit_pips", "stop_loss_pips"}
ACCEPTANCE_KEYS = {
    "min_samples",
    "min_active_days",
    "max_daily_sample_share",
    "min_positive_day_rate",
    "min_directional_hit_wilson95_lower",
    "min_avg_final_pips_exclusive",
    "min_avg_realized_r_exclusive",
    "min_fixed_exit_avg_realized_pips_exclusive",
    "min_fixed_exit_win_wilson95_lower",
    "min_fixed_exit_profit_factor",
}
LOCK_KEYS = {
    "schema_version",
    "contract_kind",
    "prepared_at_utc",
    "semantics_version",
    "evaluator",
    "training_evidence",
    "source_anchor",
    "truth_acquisition",
    "cohort",
    "candidate",
    "live_permission_granted",
    "lock_sha256",
}
EVALUATOR_KEYS = {
    "replay_path",
    "replay_sha256",
    "forward_path",
    "forward_sha256",
    "python_path",
    "python_sha256",
    "semantic_dependency_sha256s",
}
TRAINING_KEYS = {
    "report_ref",
    "report_sha256",
    "experiment_id",
    "forecast_to_utc_exclusive",
    "max_evaluated_horizon_min",
}
SOURCE_KEYS = {
    "forecast_history_ref",
    "prefix_bytes",
    "prefix_sha256",
    "emission_receipts_ref",
    "emission_receipts_prefix_bytes",
    "emission_receipts_prefix_sha256",
    "emission_receipts_last_sha256",
}
TRUTH_KEYS = {
    "history_root",
    "receipt_ref",
    "receipt_prefix_bytes",
    "receipt_prefix_sha256",
    "receipt_last_sha256",
    "fetch_script_path",
    "fetch_script_sha256",
    "granularity",
    "price_component",
    "base_url",
}
PRODUCTION_OANDA_BASE_URL = "https://api-fxtrade.oanda.com"
TRUTH_RECEIPT_FILE = "truth_acquisition_receipts.jsonl"
TRUTH_RECEIPT_SCHEMA = "QR_OANDA_TRUTH_ACQUISITION_RECEIPT_V1"
TRUTH_MARKER_KIND = "QR_FORECAST_FORWARD_TRUTH_ACQUISITION_V1"
TRUTH_MARKER_GLOB = ".qr_forward_truth_acquisition_*.json"
TRUTH_MARKER_KEYS = {
    "schema_version",
    "contract_kind",
    "lock_sha256",
    "command",
    "command_sha256",
    "environment_contract",
    "environment_contract_sha256",
    "receipt_ledger_sha256",
    "receipt_rows",
    "receipt_last_sha256",
    "artifact_files",
    "marker_sha256",
}
TRUTH_RECEIPT_KEYS = {
    "schema_version",
    "sequence",
    "recorded_at_utc",
    "output_root",
    "candle_path",
    "candle_sha256",
    "pair",
    "granularity",
    "price_component",
    "window",
    "rows",
    "fetch_script_path",
    "fetch_script_sha256",
    "previous_receipt_sha256",
    "receipt_sha256",
}
COHORT_KEYS = {
    "mode",
    "forecast_from_utc_inclusive",
    "forecast_to_utc_exclusive",
    "matures_at_utc",
    "pair_scope",
    "granularity",
    "price_component",
    "independent_non_overlap_per_pair",
    "same_bar_tp_sl",
}
PROOF_HARD_FLOORS = {
    "min_samples": 30,
    "min_active_days": 5,
    "max_daily_sample_share": 0.35,
    "min_positive_day_rate": 0.60,
    "min_directional_hit_wilson95_lower": 0.55,
    "min_avg_final_pips_exclusive": 0.0,
    "min_avg_realized_r_exclusive": 0.0,
    "min_fixed_exit_avg_realized_pips_exclusive": 0.5,
    "min_fixed_exit_win_wilson95_lower": 0.50,
    "min_fixed_exit_profit_factor": 1.5,
}
EMISSION_RECEIPT_KEYS = {
    "schema_version",
    "sequence",
    "operation",
    "recorded_at_utc",
    "forecast_timestamp_utc",
    "cycle_id",
    "pair",
    "forecast_row_sha256",
    "previous_receipt_sha256",
    "receipt_sha256",
}
RESULT_REGISTRY_KEYS = {
    "schema_version",
    "sequence",
    "registered_at_utc",
    "lock_sha256",
    "result_sha256",
    "previous_registry_sha256",
    "registry_sha256",
}
RESULT_HEAD_KEYS = {
    "schema_version",
    "contract_kind",
    "lock_sha256",
    "current_result_sha256",
    "material_result_sha256s",
    "result_registry_last_sha256",
    "proof_eligible",
    "live_permission_granted",
    "head_sha256",
}
RESULT_KEYS = {
    "schema_version",
    "contract_kind",
    "status",
    "semantics_version",
    "lock_ref",
    "lock_sha256",
    "experiment_id",
    "cohort",
    "candidate",
    "validation",
    "acceptance_checks",
    "proof_eligible",
    "proof_blockers",
    "authoritative_head_required",
    "live_permission_granted",
    "result_sha256",
}
COHORT_REGISTRY_SCHEMA = "QR_FORECAST_FORWARD_COHORT_REGISTRY_V1"
COHORT_REGISTRY_KEYS = {
    "schema_version",
    "sequence",
    "event_kind",
    "registered_at_utc",
    "lock_sha256",
    "pair",
    "from_utc",
    "to_utc",
    "result_sha256",
    "result_registry_sha256",
    "previous_event_sha256",
    "event_sha256",
}


def main() -> int:
    args = _parse_args()
    if args.command == "lock":
        result = create_lock(
            candidate_path=args.candidate_contract,
            training_report_path=args.training_report,
            forecast_history_path=args.forecast_history,
            holdout_from=_required_time(args.holdout_from, "--holdout-from"),
            holdout_to=_required_time(args.holdout_to, "--holdout-to"),
            granularity=args.granularity,
            truth_roots=args.truth_root,
            output_dir=args.output_dir,
        )
    else:
        result = evaluate_lock(
            lock_path=args.lock,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def create_lock(
    *,
    candidate_path: Path,
    training_report_path: Path,
    forecast_history_path: Path,
    holdout_from: datetime,
    holdout_to: datetime,
    granularity: str,
    truth_roots: Sequence[Path],
    output_dir: Path,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if not now < holdout_from < holdout_to:
        raise ValueError("holdout must be a future fixed [from,to) window")
    candidate = _load_json_object(candidate_path)
    _validate_candidate(candidate)
    training = _load_json_object(training_report_path)
    training_to = _training_forecast_to(training)
    training_max_horizon = _training_max_horizon(training)
    if (
        training_to is None
        or training_max_horizon is None
        or training_to + timedelta(minutes=training_max_horizon) > holdout_from
    ):
        raise ValueError("training truth horizon must end no later than holdout start")
    training_experiment_id = str(
        ((training.get("experiment") or {}).get("experiment_id")) or ""
    ).strip()
    if not training_experiment_id:
        raise ValueError("training report experiment_id required")
    max_horizon = float(candidate["max_horizon_min"])
    matures_at = holdout_to + timedelta(minutes=max_horizon) + _granularity_delta(granularity)
    forecast_history_path = forecast_history_path.resolve()
    prefix = forecast_history_path.read_bytes()
    _reject_future_rows_in_locked_prefix(prefix, holdout_from=holdout_from)
    receipt_path = forecast_history_path.with_name("forecast_emission_receipts.jsonl")
    if not receipt_path.exists():
        _write_text_atomic(receipt_path, "")
    receipt_prefix = receipt_path.read_bytes()
    receipt_rows = _validate_emission_receipt_chain(receipt_prefix)
    receipt_last_sha = str((receipt_rows[-1] if receipt_rows else {}).get("receipt_sha256") or "") or None
    if len(truth_roots) != 1:
        raise ValueError("exactly one dedicated truth root is required")
    truth_root = truth_roots[0].resolve()
    truth_root.mkdir(parents=True, exist_ok=True)
    if any(truth_root.iterdir()):
        raise ValueError("dedicated truth root must be empty when the holdout is locked")
    truth_receipt_path = truth_root / TRUTH_RECEIPT_FILE
    truth_receipt_prefix = b""
    fetch_script = SCRIPT_ROOT / "oanda_history_fetch.py"
    training_bytes = training_report_path.read_bytes()
    body: dict[str, Any] = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "contract_kind": LOCK_KIND,
        "prepared_at_utc": _iso(now),
        "semantics_version": SEMANTICS_VERSION,
        "evaluator": {
            "replay_path": str(Path(replay.__file__).resolve()),
            "replay_sha256": _sha256_bytes(Path(replay.__file__).read_bytes()),
            "forward_path": str(Path(__file__).resolve()),
            "forward_sha256": _sha256_bytes(Path(__file__).read_bytes()),
            "python_path": str(Path(sys.executable).resolve()),
            "python_sha256": _sha256_bytes(Path(sys.executable).resolve().read_bytes()),
            "semantic_dependency_sha256s": _semantic_dependency_sha256s(),
        },
        "training_evidence": {
            "report_ref": str(training_report_path.resolve()),
            "report_sha256": _sha256_bytes(training_bytes),
            "experiment_id": training_experiment_id,
            "forecast_to_utc_exclusive": _iso(training_to),
            "max_evaluated_horizon_min": training_max_horizon,
        },
        "source_anchor": {
            "forecast_history_ref": str(forecast_history_path),
            "prefix_bytes": len(prefix),
            "prefix_sha256": _sha256_bytes(prefix),
            "emission_receipts_ref": str(receipt_path),
            "emission_receipts_prefix_bytes": len(receipt_prefix),
            "emission_receipts_prefix_sha256": _sha256_bytes(receipt_prefix),
            "emission_receipts_last_sha256": receipt_last_sha,
        },
        "truth_acquisition": {
            "history_root": str(truth_root),
            "receipt_ref": str(truth_receipt_path),
            "receipt_prefix_bytes": len(truth_receipt_prefix),
            "receipt_prefix_sha256": _sha256_bytes(truth_receipt_prefix),
            "receipt_last_sha256": None,
            "fetch_script_path": str(fetch_script.resolve()),
            "fetch_script_sha256": _sha256_bytes(fetch_script.read_bytes()),
            "granularity": str(granularity).upper(),
            "price_component": "BA",
            "base_url": PRODUCTION_OANDA_BASE_URL,
        },
        "cohort": {
            "mode": "FIXED_TIME_NO_EARLY_STOP",
            "forecast_from_utc_inclusive": _iso(holdout_from),
            "forecast_to_utc_exclusive": _iso(holdout_to),
            "matures_at_utc": _iso(matures_at),
            "pair_scope": [candidate["pair"]],
            "granularity": str(granularity).upper(),
            "price_component": "BA",
            "independent_non_overlap_per_pair": True,
            "same_bar_tp_sl": "STOP_FIRST",
        },
        "candidate": candidate,
        "live_permission_granted": False,
    }
    digest = _content_sha256(body)
    lock = {**body, "lock_sha256": digest}
    locks_dir = output_dir / "locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    lock_path = locks_dir / f"{digest}.json"
    registry_path = output_dir / "cohort_registry.jsonl"
    _register_cohort(registry_path, lock)
    encoded = _canonical_pretty(lock)
    if lock_path.exists() and lock_path.read_text(encoding="utf-8") != encoded:
        raise ValueError("existing lock file content does not match its digest")
    _write_text_atomic(lock_path, encoded)
    return {"status": "LOCKED", "lock_path": str(lock_path), "lock_sha256": digest}


def evaluate_lock(
    *,
    lock_path: Path,
    output_dir: Path,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    lock = _load_json_object(lock_path)
    _validate_lock(lock, lock_path=lock_path)
    canonical_output_dir = lock_path.parent.parent.resolve()
    if output_dir.resolve() != canonical_output_dir:
        raise ValueError("forward output directory must match the lock registry root")
    output_dir = canonical_output_dir
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cohort = lock["cohort"]
    matures_at = _required_time(cohort["matures_at_utc"], "cohort.matures_at_utc")
    if now < matures_at:
        return {
            "status": "PENDING_NOT_MATURE",
            "matures_at_utc": _iso(matures_at),
            "lock_sha256": lock["lock_sha256"],
        }
    source = lock["source_anchor"]
    forecast_path = Path(source["forecast_history_ref"])
    source_bytes = forecast_path.read_bytes()
    prefix_bytes = int(source["prefix_bytes"])
    if len(source_bytes) < prefix_bytes:
        raise ValueError("forecast history was truncated below locked prefix")
    if _sha256_bytes(source_bytes[:prefix_bytes]) != source["prefix_sha256"]:
        raise ValueError("forecast history locked prefix changed")
    _validate_append_order(source_bytes[prefix_bytes:])
    receipts_path = Path(source["emission_receipts_ref"])
    receipt_bytes = receipts_path.read_bytes()
    receipt_prefix_bytes = int(source["emission_receipts_prefix_bytes"])
    if len(receipt_bytes) < receipt_prefix_bytes:
        raise ValueError("forecast emission receipt ledger was truncated")
    if _sha256_bytes(receipt_bytes[:receipt_prefix_bytes]) != source["emission_receipts_prefix_sha256"]:
        raise ValueError("forecast emission receipt locked prefix changed")
    receipt_rows = _validate_emission_receipt_chain(receipt_bytes)
    prefix_receipts = _validate_emission_receipt_chain(receipt_bytes[:receipt_prefix_bytes])
    prefix_last = str((prefix_receipts[-1] if prefix_receipts else {}).get("receipt_sha256") or "") or None
    if prefix_last != source.get("emission_receipts_last_sha256"):
        raise ValueError("forecast emission receipt anchor mismatch")

    candidate = lock["candidate"]
    forecast_from = _required_time(cohort["forecast_from_utc_inclusive"], "cohort.from")
    forecast_to = _required_time(cohort["forecast_to_utc_exclusive"], "cohort.to")
    confidence = candidate["confidence_policy"]
    rows, load_stats = replay._load_forecasts(
        forecast_path,
        pairs={candidate["pair"]},
        time_from=forecast_from,
        time_to=forecast_to,
        min_confidence=float(confidence["minimum"]),
        confidence_field=str(confidence["field"]),
    )
    direction = str(candidate["direction"])
    direction_rows = [row for row in rows if direction == "ANY" or row.direction == direction]
    context_missing_or_invalid = sum(1 for row in direction_rows if row.technical_context_status != "VALID")
    context_incomplete = sum(
        1
        for row in direction_rows
        if row.technical_context_status == "VALID"
        and not bool(((row.technical_context_v1 or {}).get("completeness") or {}).get("complete"))
    )
    selected = [
        row
        for row in direction_rows
        if row.technical_context_status == "VALID"
        and bool(((row.technical_context_v1 or {}).get("completeness") or {}).get("complete"))
        and row.horizon_min <= float(candidate["max_horizon_min"])
        and _selector_matches(row, candidate["technical_selector"])
    ]
    selected, independence = replay._select_independent_forecasts(selected)
    receipt_blockers = _forecast_receipt_blockers(
        forecast_path=forecast_path,
        selected=selected,
        receipts=receipt_rows,
        pair=str(candidate["pair"]),
        forecast_from=forecast_from,
        forecast_to=forecast_to,
    )
    _ensure_locked_truth_acquisition(lock, observed_at_utc=now)
    truth_observed_at = max(now, datetime.now(timezone.utc))
    verified_truth, truth_receipt_blockers, truth_receipt_stats = _verified_truth_files(
        lock,
        observed_at_utc=truth_observed_at,
    )
    with tempfile.TemporaryDirectory(prefix="qr-forward-truth-") as isolated:
        isolated_root = Path(isolated)
        _materialize_verified_truth(verified_truth, isolated_root)
        candles, candle_stats = replay._load_candles(
            [isolated_root],
            granularity=cohort["granularity"],
            windows_by_pair=replay._forecast_truth_windows(selected),
        )
    results, score_stats, no_market, pending = replay._score_forecasts(
        selected,
        candles,
        now_utc=now,
        granularity=cohort["granularity"],
    )

    exit_policy = candidate["exit_policy"]
    tp = float(exit_policy["take_profit_pips"])
    sl = float(exit_policy["stop_loss_pips"])
    realized = [replay._simulate_exit(row, take_profit_pips=tp, stop_loss_pips=sl) for row in results]
    fixed_exit = replay._exit_summary(realized, take_profit_pips=tp, stop_loss_pips=sl)
    fixed_wins = sum(1 for row in realized if float(row["pips"]) > 0.0)
    fixed_wilson_lower, fixed_wilson_upper = replay._wilson_interval(fixed_wins, len(realized))
    fixed_exit["win_wilson95_lower"] = fixed_wilson_lower
    fixed_exit["win_wilson95_upper"] = fixed_wilson_upper
    directional = replay._summary(results)
    daily = replay._daily_exit_stability(results, take_profit_pips=tp, stop_loss_pips=sl)
    blockers = _integrity_blockers(
        load_stats=load_stats,
        candle_stats=candle_stats,
        score_stats=score_stats,
        selected_rows=len(selected),
        evaluated_rows=len(results),
        no_market_rows=len(no_market),
        pending_rows=len(pending),
        context_missing_or_invalid=context_missing_or_invalid,
        context_incomplete=context_incomplete,
    )
    blockers.extend(receipt_blockers)
    blockers.extend(truth_receipt_blockers)
    # Local hash chains detect ordinary drift but cannot prove their latest tip
    # survived a full-filesystem rollback. Until a remote/append-only anchor is
    # configured, this evaluator collects forward evidence but cannot promote
    # it to proof eligibility.
    blockers.append("EXTERNAL_MONOTONIC_ANCHOR_NOT_CONFIGURED")
    checks = _acceptance_checks(
        candidate["acceptance"],
        directional=directional,
        fixed_exit=fixed_exit,
        daily=daily,
    )
    blockers.extend(str(check["blocker"]) for check in checks if not check["passed"])
    lock_sha = str(lock["lock_sha256"])
    report_body: dict[str, Any] = _json_safe({
        "schema_version": RESULT_SCHEMA_VERSION,
        "contract_kind": RESULT_KIND,
        "status": "COMPLETE",
        "semantics_version": SEMANTICS_VERSION,
        "lock_ref": f"{lock_path.resolve()}#sha256={lock_sha}",
        "lock_sha256": lock_sha,
        "experiment_id": _sha256_text(
            "|".join(
                (
                    RESULT_KIND,
                    lock_sha,
                    replay._forecast_rows_digest(selected),
                    replay._truth_candles_digest(candles),
                )
            )
        ),
        "cohort": {
            "forecast_from_utc_inclusive": _iso(forecast_from),
            "forecast_to_utc_exclusive": _iso(forecast_to),
            "matures_at_utc": _iso(matures_at),
            "forecast_rows_sha256": replay._forecast_rows_digest(selected),
            "truth_candles_sha256": replay._truth_candles_digest(candles),
            "selected_rows": len(selected),
            "evaluated_rows": len(results),
            "context_missing_or_invalid_rows": context_missing_or_invalid,
            "context_incomplete_rows": context_incomplete,
            "unscorable_no_market_rows": len(no_market),
            "pending_future_truth_rows": len(pending),
            "load_stats": load_stats,
            "independence": independence,
            "candle_stats": candle_stats,
            "score_stats": score_stats,
            "truth_acquisition_receipts": truth_receipt_stats,
        },
        "candidate": candidate,
        "validation": {
            "directional": directional,
            "fixed_exit": fixed_exit,
            "daily_stability": daily,
        },
        "acceptance_checks": checks,
        "proof_eligible": not blockers,
        "proof_blockers": sorted(set(blockers)),
        "authoritative_head_required": True,
        "live_permission_granted": False,
    })
    result_sha = _content_sha256(report_body)
    report = {**report_body, "result_sha256": result_sha}
    result_dir = output_dir / "forward_results" / lock_sha
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{result_sha}.json"
    encoded = _canonical_pretty(report)
    if result_path.exists() and result_path.read_text(encoding="utf-8") != encoded:
        raise ValueError("existing result file content does not match its digest")
    prior_result_shas = _result_registry_shas(
        result_dir / "result_registry.jsonl",
        lock_sha256=lock_sha,
    )
    if any(item != result_sha for item in prior_result_shas):
        report["proof_eligible"] = False
        report["proof_blockers"] = sorted(set([*report["proof_blockers"], "HOLDOUT_BACKFILL_CHANGED_RESULT"]))
        report_body = {key: value for key, value in report.items() if key != "result_sha256"}
        result_sha = _content_sha256(report_body)
        report["result_sha256"] = result_sha
        result_path = result_dir / f"{result_sha}.json"
        encoded = _canonical_pretty(report)
    _write_text_atomic(result_path, encoded)
    _register_result(
        result_dir / "result_registry.jsonl",
        lock_sha256=lock_sha,
        result_sha256=result_sha,
    )
    head_path = _write_result_head(
        result_dir=result_dir,
        lock_sha256=lock_sha,
        current_result_sha256=result_sha,
        current_report=report,
    )
    _write_text_atomic(output_dir / "latest.json", encoded)
    return {
        "status": "COMPLETE",
        "result_path": str(result_path),
        "result_sha256": result_sha,
        "head_path": str(head_path),
        "proof_eligible": report["proof_eligible"],
        "proof_blockers": report["proof_blockers"],
    }


def verify_forward_result(result_path: Path) -> tuple[bool, list[str]]:
    """Verify an immutable result against the mutable authoritative lock head."""

    blockers: list[str] = []
    report = _load_json_object(result_path)
    if set(report) != RESULT_KEYS:
        blockers.append("FORWARD_RESULT_SCHEMA_INVALID")
    stored_sha = str(report.get("result_sha256") or "")
    body = {key: value for key, value in report.items() if key != "result_sha256"}
    if result_path.stem != stored_sha or _content_sha256(body) != stored_sha:
        blockers.append("FORWARD_RESULT_DIGEST_INVALID")
    if report.get("contract_kind") != RESULT_KIND:
        blockers.append("FORWARD_RESULT_CONTRACT_INVALID")
    if (
        report.get("schema_version") != RESULT_SCHEMA_VERSION
        or report.get("status") != "COMPLETE"
        or report.get("semantics_version") != SEMANTICS_VERSION
        or report.get("authoritative_head_required") is not True
        or report.get("live_permission_granted") is not False
    ):
        blockers.append("FORWARD_RESULT_CONTRACT_INVALID")
    proof_blockers = report.get("proof_blockers")
    acceptance_checks = report.get("acceptance_checks")
    if (
        proof_blockers != []
        or not isinstance(acceptance_checks, list)
        or len(acceptance_checks) != len(ACCEPTANCE_KEYS)
        or any(
            not isinstance(check, Mapping)
            or check.get("passed") is not True
            for check in acceptance_checks
        )
    ):
        blockers.append("FORWARD_RESULT_ACCEPTANCE_INVALID")
    lock_sha = str(report.get("lock_sha256") or "")
    lock_ref = str(report.get("lock_ref") or "")
    lock_ref_path, marker, lock_ref_sha = lock_ref.rpartition("#sha256=")
    lock_path = Path(lock_ref_path) if marker else Path()
    if not marker or lock_ref_sha != lock_sha or not lock_path.is_file():
        blockers.append("FORWARD_RESULT_LOCK_REF_INVALID")
    else:
        try:
            lock = _load_json_object(lock_path)
            _validate_lock(lock, lock_path=lock_path)
            if lock.get("lock_sha256") != lock_sha or report.get("candidate") != lock.get("candidate"):
                blockers.append("FORWARD_RESULT_LOCK_BINDING_INVALID")
            lock_cohort = _mapping(lock.get("cohort"), "cohort")
            report_cohort = _mapping(report.get("cohort"), "result.cohort")
            for result_key, lock_key in (
                ("forecast_from_utc_inclusive", "forecast_from_utc_inclusive"),
                ("forecast_to_utc_exclusive", "forecast_to_utc_exclusive"),
                ("matures_at_utc", "matures_at_utc"),
            ):
                if report_cohort.get(result_key) != lock_cohort.get(lock_key):
                    blockers.append("FORWARD_RESULT_LOCK_BINDING_INVALID")
            canonical_dir = lock_path.parent.parent / "forward_results" / lock_sha
            if result_path.resolve().parent != canonical_dir.resolve():
                blockers.append("FORWARD_RESULT_NON_CANONICAL_PATH")
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            blockers.append("FORWARD_RESULT_LOCK_INVALID")
    head_path = result_path.parent / "head.json"
    if not head_path.exists():
        blockers.append("FORWARD_RESULT_HEAD_MISSING")
        return False, blockers
    head = _load_json_object(head_path)
    head_stored = str(head.get("head_sha256") or "")
    head_body = {key: value for key, value in head.items() if key != "head_sha256"}
    if (
        set(head) != RESULT_HEAD_KEYS
        or head.get("contract_kind") != RESULT_HEAD_KIND
        or head.get("live_permission_granted") is not False
        or head.get("lock_sha256") != lock_sha
        or _content_sha256(head_body) != head_stored
    ):
        blockers.append("FORWARD_RESULT_HEAD_INVALID")
    if head.get("current_result_sha256") != stored_sha:
        blockers.append("FORWARD_RESULT_NOT_AUTHORITATIVE_HEAD")
    registry_path = result_path.parent / "result_registry.jsonl"
    try:
        registry_rows = _validate_result_registry(registry_path, lock_sha256=lock_sha)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        registry_rows = []
        blockers.append("FORWARD_RESULT_REGISTRY_INVALID")
    registered = [str(row["result_sha256"]) for row in registry_rows]
    try:
        global_results = _global_result_registry_rows(
            result_path.parent.parent.parent / "cohort_registry.jsonl",
            lock_sha256=lock_sha,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        global_results = []
        blockers.append("FORWARD_RESULT_GLOBAL_REGISTRY_INVALID")
    actual_files = sorted(
        path.stem
        for path in result_path.parent.glob("*.json")
        if path.name != "head.json"
    )
    material = head.get("material_result_sha256s")
    registry_last = str((registry_rows[-1] if registry_rows else {}).get("registry_sha256") or "") or None
    if (
        not isinstance(material, list)
        or material != registered
        or sorted(registered) != actual_files
        or head.get("result_registry_last_sha256") != registry_last
        or registered != [stored_sha]
        or [str(row["result_sha256"]) for row in global_results] != registered
        or [str(row["result_registry_sha256"]) for row in global_results]
        != [str(row["registry_sha256"]) for row in registry_rows]
    ):
        blockers.append("FORWARD_RESULT_MULTIPLE_MATERIAL_RESULTS")
    if head.get("proof_eligible") is not True or report.get("proof_eligible") is not True:
        blockers.append("FORWARD_RESULT_NOT_PROOF_ELIGIBLE")
    if not blockers and marker and lock_path.is_file():
        try:
            recomputed = evaluate_lock(
                lock_path=lock_path,
                output_dir=lock_path.parent.parent,
            )
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
            blockers.append("FORWARD_RESULT_RECOMPUTE_FAILED")
        else:
            if (
                recomputed.get("status") != "COMPLETE"
                or recomputed.get("result_sha256") != stored_sha
                or recomputed.get("proof_eligible") is not True
            ):
                blockers.append("FORWARD_RESULT_RECOMPUTE_MISMATCH")
    return not blockers, sorted(set(blockers))


def _write_result_head(
    *,
    result_dir: Path,
    lock_sha256: str,
    current_result_sha256: str,
    current_report: Mapping[str, Any],
) -> Path:
    registry_rows = _validate_result_registry(
        result_dir / "result_registry.jsonl",
        lock_sha256=lock_sha256,
    )
    material = [str(row["result_sha256"]) for row in registry_rows]
    registry_last = str((registry_rows[-1] if registry_rows else {}).get("registry_sha256") or "") or None
    body = {
        "schema_version": 1,
        "contract_kind": RESULT_HEAD_KIND,
        "lock_sha256": lock_sha256,
        "current_result_sha256": current_result_sha256,
        "material_result_sha256s": material,
        "result_registry_last_sha256": registry_last,
        "proof_eligible": (
            len(material) == 1
            and material == [current_result_sha256]
            and current_report.get("proof_eligible") is True
        ),
        "live_permission_granted": False,
    }
    head = {**body, "head_sha256": _content_sha256(body)}
    path = result_dir / "head.json"
    _write_text_atomic(path, _canonical_pretty(head))
    return path


def _register_result(path: Path, *, lock_sha256: str, result_sha256: str) -> None:
    """Append one canonical result identity; never remove superseded results."""

    path.parent.mkdir(parents=True, exist_ok=True)
    registry_sha256: str | None = None
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        rows = _validate_result_registry_bytes(
            handle.read().encode("utf-8"),
            lock_sha256=lock_sha256,
        )
        existing = next(
            (row for row in rows if row["result_sha256"] == result_sha256),
            None,
        )
        if existing is not None:
            registry_sha256 = str(existing["registry_sha256"])
        else:
            body = {
                "schema_version": RESULT_REGISTRY_KIND,
                "sequence": len(rows) + 1,
                "registered_at_utc": _iso(datetime.now(timezone.utc)),
                "lock_sha256": lock_sha256,
                "result_sha256": result_sha256,
                "previous_registry_sha256": (
                    str(rows[-1]["registry_sha256"]) if rows else None
                ),
            }
            row = {**body, "registry_sha256": _content_sha256(body)}
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            registry_sha256 = str(row["registry_sha256"])
    if registry_sha256 is None:
        raise ValueError("forward result registry write failed")
    _register_global_result_event(
        path.parents[2] / "cohort_registry.jsonl",
        lock_sha256=lock_sha256,
        result_sha256=result_sha256,
        result_registry_sha256=registry_sha256,
    )


def _result_registry_shas(path: Path, *, lock_sha256: str) -> list[str]:
    return [
        str(row["result_sha256"])
        for row in _validate_result_registry(path, lock_sha256=lock_sha256)
    ]


def _validate_result_registry(
    path: Path,
    *,
    lock_sha256: str,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _validate_result_registry_bytes(path.read_bytes(), lock_sha256=lock_sha256)


def _validate_result_registry_bytes(
    payload: bytes,
    *,
    lock_sha256: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    seen_results: set[str] = set()
    for raw in payload.decode("utf-8").splitlines():
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != RESULT_REGISTRY_KEYS:
            raise ValueError("forward result registry schema invalid")
        if value.get("schema_version") != RESULT_REGISTRY_KIND:
            raise ValueError("forward result registry version invalid")
        if value.get("sequence") != len(rows) + 1:
            raise ValueError("forward result registry sequence invalid")
        if value.get("lock_sha256") != lock_sha256:
            raise ValueError("forward result registry lock mismatch")
        if value.get("previous_registry_sha256") != previous:
            raise ValueError("forward result registry chain broken")
        if _required_time(value.get("registered_at_utc"), "registered_at_utc") is None:
            raise ValueError("forward result registry time invalid")
        result_sha = str(value.get("result_sha256") or "")
        if len(result_sha) != 64 or result_sha in seen_results:
            raise ValueError("forward result registry result invalid")
        body = {key: item for key, item in value.items() if key != "registry_sha256"}
        expected = _content_sha256(body)
        if value.get("registry_sha256") != expected:
            raise ValueError("forward result registry digest invalid")
        previous = expected
        seen_results.add(result_sha)
        rows.append(value)
    return rows


def _validate_candidate(candidate: Mapping[str, Any]) -> None:
    _strict_keys(candidate, CANDIDATE_KEYS, "candidate")
    if candidate.get("contract_kind") != CANDIDATE_KIND:
        raise ValueError("candidate contract_kind invalid")
    if not str(candidate.get("candidate_id") or "").strip():
        raise ValueError("candidate_id required")
    pair = str(candidate.get("pair") or "").upper()
    if "_" not in pair or pair != candidate.get("pair"):
        raise ValueError("candidate pair must be canonical uppercase instrument")
    if candidate.get("direction") not in {"UP", "DOWN", "ANY"}:
        raise ValueError("candidate direction must be UP, DOWN, or ANY")
    confidence = _mapping(candidate.get("confidence_policy"), "confidence_policy")
    _strict_keys(confidence, CONFIDENCE_KEYS, "confidence_policy")
    if confidence.get("field") not in {"calibrated", "raw"}:
        raise ValueError("confidence field invalid")
    minimum = _typed_finite(confidence.get("minimum"), "confidence minimum")
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("confidence minimum out of range")
    selector = _mapping(candidate.get("technical_selector"), "technical_selector")
    _strict_keys(selector, set(SELECTOR_FIELDS), "technical_selector")
    for key in SELECTOR_FIELDS:
        if not str(selector.get(key) or "").strip():
            raise ValueError(f"technical selector {key} required")
    if _typed_finite(candidate.get("max_horizon_min"), "max_horizon_min") <= 0.0:
        raise ValueError("max_horizon_min must be positive")
    exit_policy = _mapping(candidate.get("exit_policy"), "exit_policy")
    _strict_keys(exit_policy, EXIT_KEYS, "exit_policy")
    for key in EXIT_KEYS:
        if _typed_finite(exit_policy.get(key), key) <= 0.0:
            raise ValueError(f"{key} must be positive")
    acceptance = _mapping(candidate.get("acceptance"), "acceptance")
    _strict_keys(acceptance, ACCEPTANCE_KEYS, "acceptance")
    for key in ACCEPTANCE_KEYS:
        _typed_finite(acceptance.get(key), key)
    for key in ("min_samples", "min_active_days"):
        value = acceptance.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in (
        "max_daily_sample_share",
        "min_positive_day_rate",
        "min_directional_hit_wilson95_lower",
        "min_fixed_exit_win_wilson95_lower",
    ):
        value = float(acceptance[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be between zero and one")
    for key, floor in PROOF_HARD_FLOORS.items():
        value = float(acceptance[key])
        if key == "max_daily_sample_share":
            if value > float(floor):
                raise ValueError(f"{key} is weaker than proof hard floor")
        elif value < float(floor):
            raise ValueError(f"{key} is weaker than proof hard floor")


def _validate_lock(lock: Mapping[str, Any], *, lock_path: Path) -> None:
    _strict_keys(lock, LOCK_KEYS, "lock")
    if lock.get("schema_version") != LOCK_SCHEMA_VERSION or lock.get("contract_kind") != LOCK_KIND:
        raise ValueError("lock schema invalid")
    if lock.get("semantics_version") != SEMANTICS_VERSION:
        raise ValueError("lock semantics invalid")
    stored = str(lock.get("lock_sha256") or "")
    body = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if stored != _content_sha256(body) or lock_path.stem != stored:
        raise ValueError("lock digest mismatch")
    evaluator = _mapping(lock.get("evaluator"), "evaluator")
    _strict_keys(evaluator, EVALUATOR_KEYS, "evaluator")
    if _sha256_bytes(Path(evaluator["replay_path"]).read_bytes()) != evaluator["replay_sha256"]:
        raise ValueError("replay evaluator changed after lock")
    if _sha256_bytes(Path(evaluator["forward_path"]).read_bytes()) != evaluator["forward_sha256"]:
        raise ValueError("forward evaluator changed after lock")
    python_path = Path(str(evaluator.get("python_path") or ""))
    if (
        python_path != Path(sys.executable).resolve()
        or _sha256_bytes(python_path.read_bytes()) != evaluator.get("python_sha256")
    ):
        raise ValueError("forward Python runtime changed after lock")
    dependencies = evaluator.get("semantic_dependency_sha256s")
    if dependencies != _semantic_dependency_sha256s():
        raise ValueError("forward semantic dependencies changed after lock")
    training = _mapping(lock.get("training_evidence"), "training_evidence")
    _strict_keys(training, TRAINING_KEYS, "training_evidence")
    if _sha256_bytes(Path(training["report_ref"]).read_bytes()) != training["report_sha256"]:
        raise ValueError("training report changed after lock")
    source = _mapping(lock.get("source_anchor"), "source_anchor")
    _strict_keys(source, SOURCE_KEYS, "source_anchor")
    truth = _mapping(lock.get("truth_acquisition"), "truth_acquisition")
    _strict_keys(truth, TRUTH_KEYS, "truth_acquisition")
    truth_root = Path(str(truth.get("history_root") or ""))
    if not truth_root.is_absolute() or str(truth_root.resolve()) != str(truth_root):
        raise ValueError("truth acquisition root must be an absolute resolved path")
    if not truth_root.is_dir():
        raise ValueError("truth acquisition root missing")
    receipt_ref = Path(str(truth.get("receipt_ref") or ""))
    if receipt_ref != truth_root / TRUTH_RECEIPT_FILE:
        raise ValueError("truth acquisition receipt path is outside its locked root")
    prefix_bytes = truth.get("receipt_prefix_bytes")
    if isinstance(prefix_bytes, bool) or not isinstance(prefix_bytes, int) or prefix_bytes < 0:
        raise ValueError("truth acquisition receipt prefix size invalid")
    prefix_sha = str(truth.get("receipt_prefix_sha256") or "")
    if len(prefix_sha) != 64:
        raise ValueError("truth acquisition receipt prefix digest invalid")
    last_sha = truth.get("receipt_last_sha256")
    if last_sha is not None and (not isinstance(last_sha, str) or len(last_sha) != 64):
        raise ValueError("truth acquisition receipt chain anchor invalid")
    if (
        truth.get("price_component") != "BA"
        or truth.get("base_url") != PRODUCTION_OANDA_BASE_URL
    ):
        raise ValueError("truth acquisition price component invalid")
    if _sha256_bytes(Path(truth["fetch_script_path"]).read_bytes()) != truth["fetch_script_sha256"]:
        raise ValueError("truth acquisition script changed after lock")
    cohort = _mapping(lock.get("cohort"), "cohort")
    _strict_keys(cohort, COHORT_KEYS, "cohort")
    if cohort.get("mode") != "FIXED_TIME_NO_EARLY_STOP" or cohort.get("price_component") != "BA":
        raise ValueError("cohort mode invalid")
    prepared = _required_time(lock.get("prepared_at_utc"), "prepared_at_utc")
    start = _required_time(cohort.get("forecast_from_utc_inclusive"), "cohort.from")
    end = _required_time(cohort.get("forecast_to_utc_exclusive"), "cohort.to")
    if not prepared < start < end:
        raise ValueError("lock was not prepared before cohort")
    candidate = _mapping(lock.get("candidate"), "candidate")
    _validate_candidate(candidate)
    expected_maturity = (
        end
        + timedelta(minutes=float(candidate["max_horizon_min"]))
        + _granularity_delta(str(cohort["granularity"]))
    )
    maturity = _required_time(cohort.get("matures_at_utc"), "cohort.matures_at_utc")
    if maturity != expected_maturity:
        raise ValueError("cohort maturity does not match locked horizon")
    if str(truth.get("granularity")) != str(cohort.get("granularity")):
        raise ValueError("truth/cohort granularity mismatch")
    if datetime.fromtimestamp(lock_path.stat().st_mtime, tz=timezone.utc) >= start:
        raise ValueError("lock file was not materialized before cohort start")
    _verify_registry_membership(lock_path.parent.parent / "cohort_registry.jsonl", lock)
    if lock.get("live_permission_granted") is not False:
        raise ValueError("forward lock can never grant live permission")


def _ensure_locked_truth_acquisition(
    lock: Mapping[str, Any],
    *,
    observed_at_utc: datetime,
) -> Path:
    """Fetch the locked truth window once, then verify its immutable marker."""

    truth = _mapping(lock.get("truth_acquisition"), "truth_acquisition")
    root = Path(str(truth["history_root"]))
    command = _locked_truth_fetch_command(lock)
    lock_file = root.parent / f".{root.name}.forward-truth-fetch.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        markers = sorted(root.glob(TRUTH_MARKER_GLOB))
        if markers:
            if len(markers) != 1:
                raise ValueError("truth acquisition marker count invalid")
            _validate_truth_acquisition_marker(
                markers[0],
                lock=lock,
                command=command,
                observed_at_utc=observed_at_utc,
            )
            return markers[0]
        if any(root.iterdir()):
            raise ValueError("dedicated truth root was prepopulated before evaluator acquisition")

        _run_locked_truth_fetch_subprocess(command)
        verification_time = max(observed_at_utc, datetime.now(timezone.utc))
        verified, blockers, _stats = _verified_truth_files(
            lock,
            observed_at_utc=verification_time,
        )
        if blockers or len(verified) != 1:
            reason = ",".join(sorted(set(blockers))) or "VERIFIED_FILE_COUNT_INVALID"
            raise ValueError(f"evaluator truth acquisition was not verifiable: {reason}")
        receipt_path = Path(str(truth["receipt_ref"]))
        receipt_bytes = receipt_path.read_bytes()
        receipts = _validate_truth_acquisition_receipt_chain(receipt_bytes)
        _validate_locked_truth_receipts(lock, receipts)
        artifacts = _truth_artifact_snapshot(root)
        environment_contract = _locked_truth_environment_contract()
        body: dict[str, Any] = {
            "schema_version": 1,
            "contract_kind": TRUTH_MARKER_KIND,
            "lock_sha256": str(lock["lock_sha256"]),
            "command": command,
            "command_sha256": _sha256_bytes(_canonical_json_bytes(command)),
            "environment_contract": environment_contract,
            "environment_contract_sha256": _content_sha256(environment_contract),
            "receipt_ledger_sha256": _sha256_bytes(receipt_bytes),
            "receipt_rows": len(receipts),
            "receipt_last_sha256": receipts[-1]["receipt_sha256"],
            "artifact_files": artifacts,
        }
        marker_sha = _content_sha256(body)
        marker = {**body, "marker_sha256": marker_sha}
        marker_path = root / f".qr_forward_truth_acquisition_{marker_sha}.json"
        _write_text_atomic(marker_path, _canonical_pretty(marker))
        return marker_path


def _run_locked_truth_fetch_subprocess(command: Sequence[str]) -> None:
    """Run only the read-only OANDA history CLI frozen by the holdout lock."""

    subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        env=_locked_truth_subprocess_env(),
    )


def _locked_truth_environment_contract() -> dict[str, Any]:
    return {
        "oanda_base_url": PRODUCTION_OANDA_BASE_URL,
        "pythonpath_inherited": False,
        "oanda_env_file_override_inherited": False,
        "allowed_inherited_environment": [
            "HOME",
            "LANG",
            "LC_ALL",
            "PATH",
            "SSL_CERT_DIR",
            "SSL_CERT_FILE",
            "TMPDIR",
            "QR_OANDA_ACCOUNT_ID",
            "QR_OANDA_TOKEN",
        ],
    }


def _locked_truth_subprocess_env() -> dict[str, str]:
    contract = _locked_truth_environment_contract()
    allowed = set(contract["allowed_inherited_environment"])
    env = {
        key: value
        for key, value in os.environ.items()
        if key in allowed and value
    }
    env["QR_OANDA_BASE_URL"] = PRODUCTION_OANDA_BASE_URL
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("QR_OANDA_ENV_FILE", None)
    env.pop("QR_OANDA_HTTP_TIMEOUT_SECONDS", None)
    return env


def _locked_truth_fetch_command(lock: Mapping[str, Any]) -> list[str]:
    truth = _mapping(lock.get("truth_acquisition"), "truth_acquisition")
    cohort = _mapping(lock.get("cohort"), "cohort")
    return [
        str(Path(sys.executable).resolve()),
        str(truth["fetch_script_path"]),
        "--pairs",
        str(lock["candidate"]["pair"]),
        "--granularities",
        str(cohort["granularity"]),
        "--price",
        "BA",
        "--from",
        str(cohort["forecast_from_utc_inclusive"]),
        "--to",
        str(cohort["matures_at_utc"]),
        "--output-dir",
        str(truth["history_root"]),
        "--max-candles-per-request",
        "4500",
        "--sleep-seconds",
        "0.2",
        "--retries",
        "3",
        "--compress",
    ]


def _validate_truth_acquisition_marker(
    marker_path: Path,
    *,
    lock: Mapping[str, Any],
    command: Sequence[str],
    observed_at_utc: datetime,
) -> None:
    marker = _load_json_object(marker_path)
    _strict_keys(marker, TRUTH_MARKER_KEYS, "truth acquisition marker")
    stored = str(marker.get("marker_sha256") or "")
    body = {key: value for key, value in marker.items() if key != "marker_sha256"}
    if (
        marker.get("schema_version") != 1
        or marker.get("contract_kind") != TRUTH_MARKER_KIND
        or stored != _content_sha256(body)
        or marker_path.name != f".qr_forward_truth_acquisition_{stored}.json"
    ):
        raise ValueError("truth acquisition marker digest mismatch")
    if marker.get("lock_sha256") != lock.get("lock_sha256"):
        raise ValueError("truth acquisition marker lock mismatch")
    expected_command = list(command)
    if marker.get("command") != expected_command:
        raise ValueError("truth acquisition command contract mismatch")
    expected_command_sha = _sha256_bytes(_canonical_json_bytes(expected_command))
    if marker.get("command_sha256") != expected_command_sha:
        raise ValueError("truth acquisition command digest mismatch")
    environment_contract = _locked_truth_environment_contract()
    if (
        marker.get("environment_contract") != environment_contract
        or marker.get("environment_contract_sha256")
        != _content_sha256(environment_contract)
    ):
        raise ValueError("truth acquisition environment contract mismatch")

    truth = _mapping(lock.get("truth_acquisition"), "truth_acquisition")
    receipt_bytes = Path(str(truth["receipt_ref"])).read_bytes()
    if marker.get("receipt_ledger_sha256") != _sha256_bytes(receipt_bytes):
        raise ValueError("truth acquisition receipt ledger changed after acquisition")
    receipts = _validate_truth_acquisition_receipt_chain(receipt_bytes)
    if (
        marker.get("receipt_rows") != len(receipts)
        or not receipts
        or marker.get("receipt_last_sha256") != receipts[-1]["receipt_sha256"]
    ):
        raise ValueError("truth acquisition receipt marker mismatch")
    _validate_locked_truth_receipts(lock, receipts)
    artifacts = _truth_artifact_snapshot(Path(str(truth["history_root"])))
    if marker.get("artifact_files") != artifacts:
        raise ValueError("truth acquisition files changed after acquisition")
    verified, blockers, _stats = _verified_truth_files(
        lock,
        observed_at_utc=observed_at_utc,
    )
    if blockers or len(verified) != 1:
        reason = ",".join(sorted(set(blockers))) or "VERIFIED_FILE_COUNT_INVALID"
        raise ValueError(f"marked truth acquisition is not verifiable: {reason}")


def _validate_locked_truth_receipts(
    lock: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> None:
    truth = _mapping(lock.get("truth_acquisition"), "truth_acquisition")
    cohort = _mapping(lock.get("cohort"), "cohort")
    if len(receipts) != 1:
        raise ValueError("truth acquisition must contain exactly one locked receipt")
    receipt = receipts[0]
    expected_window = {
        "from_utc": str(cohort["forecast_from_utc_inclusive"]),
        "to_utc": str(cohort["matures_at_utc"]),
    }
    if (
        receipt.get("output_root") != truth.get("history_root")
        or receipt.get("pair") != lock["candidate"]["pair"]
        or receipt.get("granularity") != cohort["granularity"]
        or receipt.get("price_component") != "BA"
        or receipt.get("window") != expected_window
        or receipt.get("fetch_script_path") != truth.get("fetch_script_path")
        or receipt.get("fetch_script_sha256") != truth.get("fetch_script_sha256")
    ):
        raise ValueError("truth acquisition receipt does not match locked command")


def _truth_artifact_snapshot(root: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("truth acquisition artifacts may not contain symlinks")
        if not path.is_file() or path.match(TRUTH_MARKER_GLOB):
            continue
        data = path.read_bytes()
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(data),
                "sha256": _sha256_bytes(data),
            }
        )
    if not artifacts:
        raise ValueError("truth acquisition produced no artifacts")
    return artifacts


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _verified_truth_files(
    lock: Mapping[str, Any],
    *,
    observed_at_utc: datetime,
) -> tuple[list[tuple[str, str, bytes]], list[str], dict[str, Any]]:
    """Return only immutable bytes proved by post-lock acquisition receipts."""

    truth = _mapping(lock.get("truth_acquisition"), "truth_acquisition")
    root = Path(str(truth["history_root"]))
    receipt_path = Path(str(truth["receipt_ref"]))
    blockers: list[str] = []
    stats: dict[str, Any] = {
        "receipt_rows": 0,
        "post_lock_receipt_rows": 0,
        "verified_files": 0,
        "unproven_files": 0,
    }
    if not receipt_path.exists():
        discovered = {
            path.resolve()
            for path in root.glob(
                f"**/*_{lock['cohort']['granularity']}_BA_*.jsonl*"
            )
            if replay._is_supported_history_file(path)
        }
        stats["unproven_files"] = len(discovered)
        missing_blockers = ["TRUTH_ACQUISITION_RECEIPT_MISSING"]
        if discovered:
            missing_blockers.append("TRUTH_ACQUISITION_UNPROVEN_FILE")
        return [], missing_blockers, stats
    receipt_bytes = receipt_path.read_bytes()
    prefix_bytes = int(truth["receipt_prefix_bytes"])
    if len(receipt_bytes) < prefix_bytes:
        return [], ["TRUTH_ACQUISITION_RECEIPT_PREFIX_TRUNCATED"], stats
    if _sha256_bytes(receipt_bytes[:prefix_bytes]) != truth["receipt_prefix_sha256"]:
        return [], ["TRUTH_ACQUISITION_RECEIPT_PREFIX_CHANGED"], stats
    try:
        prefix_rows = _validate_truth_acquisition_receipt_chain(receipt_bytes[:prefix_bytes])
        receipt_rows = _validate_truth_acquisition_receipt_chain(receipt_bytes)
    except ValueError:
        return [], ["TRUTH_ACQUISITION_RECEIPT_CHAIN_INVALID"], stats
    prefix_last = str(prefix_rows[-1]["receipt_sha256"]) if prefix_rows else None
    if prefix_last != truth.get("receipt_last_sha256"):
        return [], ["TRUTH_ACQUISITION_RECEIPT_ANCHOR_MISMATCH"], stats

    stats["receipt_rows"] = len(receipt_rows)
    suffix = receipt_rows[len(prefix_rows):]
    stats["post_lock_receipt_rows"] = len(suffix)
    prepared_at = _required_time(lock.get("prepared_at_utc"), "prepared_at_utc")
    expected_root = str(root)
    expected_script = str(truth["fetch_script_path"])
    expected_script_sha = str(truth["fetch_script_sha256"])
    expected_pair = str(lock["candidate"]["pair"])
    expected_granularity = str(lock["cohort"]["granularity"])
    verified: list[tuple[str, str, bytes]] = []
    verified_paths: set[Path] = set()
    seen_receipt_paths: set[Path] = set()

    for receipt in suffix:
        recorded_at = _required_time(receipt.get("recorded_at_utc"), "truth receipt time")
        if not prepared_at < recorded_at <= observed_at_utc:
            blockers.append("TRUTH_ACQUISITION_RECEIPT_TIME_INVALID")
            continue
        mismatch = False
        if receipt.get("output_root") != expected_root:
            blockers.append("TRUTH_ACQUISITION_ROOT_MISMATCH")
            mismatch = True
        if (
            receipt.get("fetch_script_path") != expected_script
            or receipt.get("fetch_script_sha256") != expected_script_sha
        ):
            blockers.append("TRUTH_ACQUISITION_SCRIPT_MISMATCH")
            mismatch = True
        if receipt.get("pair") != expected_pair:
            blockers.append("TRUTH_ACQUISITION_PAIR_MISMATCH")
            mismatch = True
        if receipt.get("granularity") != expected_granularity:
            blockers.append("TRUTH_ACQUISITION_GRANULARITY_MISMATCH")
            mismatch = True
        if receipt.get("price_component") != "BA":
            blockers.append("TRUTH_ACQUISITION_PRICE_COMPONENT_MISMATCH")
            mismatch = True
        path = Path(str(receipt.get("candle_path") or ""))
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, ValueError):
            blockers.append("TRUTH_ACQUISITION_FILE_MISSING_OR_OUTSIDE_ROOT")
            continue
        if str(resolved) != str(path):
            blockers.append("TRUTH_ACQUISITION_FILE_PATH_NOT_RESOLVED")
            mismatch = True
        if resolved in seen_receipt_paths:
            blockers.append("TRUTH_ACQUISITION_DUPLICATE_FILE_RECEIPT")
            mismatch = True
        seen_receipt_paths.add(resolved)
        if not replay._is_supported_history_file(resolved):
            blockers.append("TRUTH_ACQUISITION_FILE_TYPE_INVALID")
            mismatch = True
        data = resolved.read_bytes()
        if _sha256_bytes(data) != receipt.get("candle_sha256"):
            blockers.append("TRUTH_ACQUISITION_FILE_HASH_MISMATCH")
            mismatch = True
        file_blockers, parsed_rows = _truth_candle_file_blockers(
            path=resolved,
            data=data,
            receipt=receipt,
        )
        blockers.extend(file_blockers)
        expected_rows = receipt.get("rows")
        if (
            isinstance(expected_rows, bool)
            or not isinstance(expected_rows, int)
            or expected_rows < 0
            or expected_rows != parsed_rows
        ):
            blockers.append("TRUTH_ACQUISITION_ROW_COUNT_MISMATCH")
            mismatch = True
        if file_blockers:
            mismatch = True
        if not mismatch:
            verified.append((expected_pair, resolved.name, data))
            verified_paths.add(resolved)

    discovered = {
        path.resolve()
        for path in root.glob(f"**/*_{expected_granularity}_BA_*.jsonl*")
        if replay._is_supported_history_file(path)
    }
    unproven = discovered - verified_paths
    if unproven:
        blockers.append("TRUTH_ACQUISITION_UNPROVEN_FILE")
    stats["verified_files"] = len(verified)
    stats["unproven_files"] = len(unproven)
    return verified, sorted(set(blockers)), stats


def _validate_truth_acquisition_receipt_chain(payload: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_sha: str | None = None
    for raw in payload.decode("utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("truth acquisition receipt ledger is malformed") from exc
        if not isinstance(item, dict) or set(item) != TRUTH_RECEIPT_KEYS:
            raise ValueError("truth acquisition receipt schema invalid")
        if item.get("schema_version") != TRUTH_RECEIPT_SCHEMA:
            raise ValueError("truth acquisition receipt version invalid")
        if item.get("sequence") != len(rows) + 1:
            raise ValueError("truth acquisition receipt sequence invalid")
        if item.get("previous_receipt_sha256") != previous_sha:
            raise ValueError("truth acquisition receipt chain broken")
        body = {key: value for key, value in item.items() if key != "receipt_sha256"}
        expected_sha = _content_sha256(body)
        if item.get("receipt_sha256") != expected_sha:
            raise ValueError("truth acquisition receipt digest mismatch")
        previous_sha = expected_sha
        rows.append(item)
    return rows


def _truth_candle_file_blockers(
    *,
    path: Path,
    data: bytes,
    receipt: Mapping[str, Any],
) -> tuple[list[str], int]:
    blockers: list[str] = []
    try:
        text = gzip.decompress(data).decode("utf-8") if path.name.endswith(".gz") else data.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return ["TRUTH_ACQUISITION_FILE_ENCODING_INVALID"], 0
    window = receipt.get("window")
    if not isinstance(window, Mapping) or set(window) != {"from_utc", "to_utc"}:
        return ["TRUTH_ACQUISITION_WINDOW_INVALID"], 0
    start = replay._parse_time(window.get("from_utc"))
    end = replay._parse_time(window.get("to_utc"))
    if start is None or end is None or start >= end:
        return ["TRUTH_ACQUISITION_WINDOW_INVALID"], 0
    rows = 0
    seen_times: set[datetime] = set()
    for raw in text.splitlines():
        if not raw.strip():
            continue
        rows += 1
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            blockers.append("TRUTH_ACQUISITION_CANDLE_ROW_INVALID")
            continue
        timestamp = replay._parse_time(item.get("time")) if isinstance(item, Mapping) else None
        if (
            not isinstance(item, Mapping)
            or item.get("pair") != receipt.get("pair")
            or item.get("granularity") != receipt.get("granularity")
            or item.get("price") != receipt.get("price_component")
            or item.get("complete") is not True
            or not isinstance(item.get("bid"), Mapping)
            or not isinstance(item.get("ask"), Mapping)
            or timestamp is None
            or not start <= timestamp <= end
            or timestamp in seen_times
        ):
            blockers.append("TRUTH_ACQUISITION_CANDLE_METADATA_MISMATCH")
            continue
        seen_times.add(timestamp)
    return sorted(set(blockers)), rows


def _materialize_verified_truth(
    verified: Sequence[tuple[str, str, bytes]],
    destination: Path,
) -> None:
    for index, (pair, name, data) in enumerate(verified, start=1):
        pair_dir = destination / pair
        pair_dir.mkdir(parents=True, exist_ok=True)
        (pair_dir / f"{index:06d}_{name}").write_bytes(data)


def _selector_matches(row: Any, selector: Mapping[str, Any]) -> bool:
    context = row.technical_context_v1 or {}
    values = {
        "technical_regime": _context_label(context, "regime", "primary"),
        "technical_atr_band": _context_label(context, "volatility", "primary_atr_band"),
        "technical_spread_band": _context_label(context, "execution", "spread_band"),
        "technical_range_location_24h": _context_label(context, "location", "range_location_24h"),
        "technical_structure_alignment": replay._technical_structure_alignment(
            context,
            forecast_direction=row.direction,
        ),
    }
    return all(str(selector[key]).upper() == "ANY" or str(selector[key]).upper() == values[key] for key in SELECTOR_FIELDS)


def _integrity_blockers(
    *,
    load_stats: Mapping[str, Any],
    candle_stats: Mapping[str, Any],
    score_stats: Mapping[str, Any],
    selected_rows: int,
    evaluated_rows: int,
    no_market_rows: int,
    pending_rows: int,
    context_missing_or_invalid: int,
    context_incomplete: int,
) -> list[str]:
    checks = {
        "NO_SELECTED_FORECASTS": selected_rows == 0,
        "SELECTED_FORECAST_TRUTH_INCOMPLETE": evaluated_rows != selected_rows,
        "TECHNICAL_CONTEXT_MISSING_OR_INVALID": context_missing_or_invalid > 0,
        "TECHNICAL_CONTEXT_INCOMPLETE": context_incomplete > 0,
        "CONFLICTING_FORECAST_ROWS": int(load_stats.get("skipped_conflicting_forecast_rows") or 0) > 0,
        "INVALID_FORECAST_ROWS": int(load_stats.get("skipped_invalid_rows") or 0) > 0,
        "CONFLICTING_PRICE_TRUTH": int(candle_stats.get("history_conflicting_candles") or 0) > 0,
        "MISSING_PAIR_PRICE_TRUTH": int(score_stats.get("skipped_no_pair_candles") or 0) > 0,
        "MISSING_PRICE_WINDOW": int(score_stats.get("skipped_no_price_window") or 0) > 0,
        "INCOMPLETE_PRICE_WINDOW": int(score_stats.get("skipped_incomplete_truth_window_rows") or 0) > 0,
        "UNSCORABLE_NO_MARKET_WINDOW": no_market_rows > 0,
        "PENDING_FUTURE_TRUTH": pending_rows > 0,
    }
    return [name for name, failed in checks.items() if failed]


def _acceptance_checks(
    acceptance: Mapping[str, Any],
    *,
    directional: Mapping[str, Any],
    fixed_exit: Mapping[str, Any],
    daily: Mapping[str, Any],
) -> list[dict[str, Any]]:
    specs = (
        ("MIN_SAMPLES", ">=", acceptance["min_samples"], directional.get("n")),
        ("MIN_ACTIVE_DAYS", ">=", acceptance["min_active_days"], daily.get("active_days")),
        ("MAX_DAILY_SAMPLE_SHARE", "<=", acceptance["max_daily_sample_share"], daily.get("max_daily_sample_share")),
        ("MIN_POSITIVE_DAY_RATE", ">=", acceptance["min_positive_day_rate"], daily.get("positive_day_rate")),
        ("MIN_DIRECTIONAL_HIT_WILSON95_LOWER", ">=", acceptance["min_directional_hit_wilson95_lower"], directional.get("hit_wilson95_lower")),
        ("MIN_AVG_FINAL_PIPS", ">", acceptance["min_avg_final_pips_exclusive"], directional.get("avg_final_pips")),
        ("MIN_AVG_REALIZED_R", ">", acceptance["min_avg_realized_r_exclusive"], directional.get("avg_realized_r")),
        ("MIN_FIXED_EXIT_AVG_REALIZED_PIPS", ">", acceptance["min_fixed_exit_avg_realized_pips_exclusive"], fixed_exit.get("avg_realized_pips")),
        ("MIN_FIXED_EXIT_WIN_WILSON95_LOWER", ">=", acceptance["min_fixed_exit_win_wilson95_lower"], fixed_exit.get("win_wilson95_lower")),
        ("MIN_FIXED_EXIT_PROFIT_FACTOR", ">=", acceptance["min_fixed_exit_profit_factor"], fixed_exit.get("profit_factor")),
    )
    out: list[dict[str, Any]] = []
    for name, operator, threshold_raw, observed_raw in specs:
        threshold = _finite(threshold_raw, name)
        observed = _optional_finite(observed_raw)
        passed = observed is not None and (
            observed >= threshold if operator == ">=" else observed <= threshold if operator == "<=" else observed > threshold
        )
        out.append(
            {
                "name": name,
                "operator": operator,
                "threshold": threshold,
                "observed": observed,
                "passed": passed,
                "blocker": f"{name}_NOT_MET",
            }
        )
    return out


def _training_forecast_to(report: Mapping[str, Any]) -> datetime | None:
    selection = report.get("selection_contract") if isinstance(report.get("selection_contract"), Mapping) else {}
    return replay._parse_time((selection or {}).get("forecast_to_utc_exclusive"))


def _training_max_horizon(report: Mapping[str, Any]) -> float | None:
    value = _optional_finite(report.get("max_evaluated_horizon_min"))
    return value if value is not None and value > 0.0 else None


def _register_cohort(path: Path, lock: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cohort = lock["cohort"]
    pair = lock["candidate"]["pair"]
    start = _required_time(cohort["forecast_from_utc_inclusive"], "cohort.from")
    end = _required_time(cohort["forecast_to_utc_exclusive"], "cohort.to")
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        rows = _validate_cohort_registry_bytes(handle.read().encode("utf-8"))
        for item in rows:
            if item.get("event_kind") != "LOCK" or str(item.get("pair")) != pair:
                continue
            old_start = _required_time(item.get("from_utc"), "registry.from")
            old_end = _required_time(item.get("to_utc"), "registry.to")
            if max(start, old_start) < min(end, old_end):
                if item.get("lock_sha256") == lock["lock_sha256"]:
                    return
                raise ValueError("forward cohort interval already consumed for pair")
        _append_cohort_registry_row(
            handle,
            rows=rows,
            event_kind="LOCK",
            lock_sha256=str(lock["lock_sha256"]),
            pair=str(pair),
            from_utc=_iso(start),
            to_utc=_iso(end),
            result_sha256=None,
            result_registry_sha256=None,
        )


def _verify_registry_membership(path: Path, lock: Mapping[str, Any]) -> None:
    if not path.exists():
        raise ValueError("forward lock registry missing")
    cohort = lock["cohort"]
    rows = _validate_cohort_registry(path)
    matches = [
        item
        for item in rows
        if item.get("event_kind") == "LOCK"
        and item.get("pair") == lock["candidate"]["pair"]
        and item.get("from_utc") == cohort["forecast_from_utc_inclusive"]
        and item.get("to_utc") == cohort["forecast_to_utc_exclusive"]
        and item.get("lock_sha256") == lock["lock_sha256"]
    ]
    if len(matches) != 1:
        raise ValueError("forward lock is not uniquely registered")
    registered_at = _required_time(matches[0]["registered_at_utc"], "registry.registered_at_utc")
    start = _required_time(cohort["forecast_from_utc_inclusive"], "cohort.from")
    if registered_at >= start:
        raise ValueError("forward lock registry event was not written before cohort start")


def _register_global_result_event(
    path: Path,
    *,
    lock_sha256: str,
    result_sha256: str,
    result_registry_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        rows = _validate_cohort_registry_bytes(handle.read().encode("utf-8"))
        lock_rows = [
            row
            for row in rows
            if row.get("event_kind") == "LOCK"
            and row.get("lock_sha256") == lock_sha256
        ]
        if len(lock_rows) != 1:
            raise ValueError("result cannot register without one canonical lock event")
        existing = [
            row
            for row in rows
            if row.get("event_kind") == "RESULT"
            and row.get("lock_sha256") == lock_sha256
            and row.get("result_sha256") == result_sha256
        ]
        if existing:
            if (
                len(existing) != 1
                or existing[0].get("result_registry_sha256")
                != result_registry_sha256
            ):
                raise ValueError("global result registry binding conflict")
            return
        lock_row = lock_rows[0]
        _append_cohort_registry_row(
            handle,
            rows=rows,
            event_kind="RESULT",
            lock_sha256=lock_sha256,
            pair=str(lock_row["pair"]),
            from_utc=str(lock_row["from_utc"]),
            to_utc=str(lock_row["to_utc"]),
            result_sha256=result_sha256,
            result_registry_sha256=result_registry_sha256,
        )


def _global_result_registry_rows(
    path: Path,
    *,
    lock_sha256: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in _validate_cohort_registry(path)
        if row.get("event_kind") == "RESULT"
        and row.get("lock_sha256") == lock_sha256
    ]


def _validate_cohort_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError("forward cohort registry missing")
    return _validate_cohort_registry_bytes(path.read_bytes())


def _validate_cohort_registry_bytes(payload: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    for raw in payload.decode("utf-8").splitlines():
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != COHORT_REGISTRY_KEYS:
            raise ValueError("cohort registry schema invalid")
        if value.get("schema_version") != COHORT_REGISTRY_SCHEMA:
            raise ValueError("cohort registry version invalid")
        sequence = value.get("sequence")
        if isinstance(sequence, bool) or sequence != len(rows) + 1:
            raise ValueError("cohort registry sequence invalid")
        if value.get("event_kind") not in {"LOCK", "RESULT"}:
            raise ValueError("cohort registry event invalid")
        if value.get("previous_event_sha256") != previous:
            raise ValueError("cohort registry chain broken")
        _required_time(value.get("registered_at_utc"), "registry.registered_at_utc")
        if value.get("event_kind") == "LOCK" and (
            value.get("result_sha256") is not None
            or value.get("result_registry_sha256") is not None
        ):
            raise ValueError("cohort lock registry payload invalid")
        if value.get("event_kind") == "RESULT" and (
            not isinstance(value.get("result_sha256"), str)
            or len(str(value.get("result_sha256"))) != 64
            or not isinstance(value.get("result_registry_sha256"), str)
            or len(str(value.get("result_registry_sha256"))) != 64
        ):
            raise ValueError("cohort result registry payload invalid")
        body = {key: item for key, item in value.items() if key != "event_sha256"}
        expected = _content_sha256(body)
        if value.get("event_sha256") != expected:
            raise ValueError("cohort registry digest invalid")
        previous = expected
        rows.append(value)
    return rows


def _append_cohort_registry_row(
    handle: Any,
    *,
    rows: Sequence[Mapping[str, Any]],
    event_kind: str,
    lock_sha256: str,
    pair: str,
    from_utc: str,
    to_utc: str,
    result_sha256: str | None,
    result_registry_sha256: str | None,
) -> dict[str, Any]:
    body = {
        "schema_version": COHORT_REGISTRY_SCHEMA,
        "sequence": len(rows) + 1,
        "event_kind": event_kind,
        "registered_at_utc": _iso(datetime.now(timezone.utc)),
        "lock_sha256": lock_sha256,
        "pair": pair,
        "from_utc": from_utc,
        "to_utc": to_utc,
        "result_sha256": result_sha256,
        "result_registry_sha256": result_registry_sha256,
        "previous_event_sha256": (
            str(rows[-1]["event_sha256"]) if rows else None
        ),
    }
    row = {**body, "event_sha256": _content_sha256(body)}
    handle.seek(0, os.SEEK_END)
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    return row


def _reject_future_rows_in_locked_prefix(prefix: bytes, *, holdout_from: datetime) -> None:
    for raw in prefix.decode("utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        timestamp = replay._parse_time(item.get("timestamp_utc")) if isinstance(item, dict) else None
        if timestamp is not None and timestamp >= holdout_from:
            raise ValueError("locked source prefix already contains holdout-period forecasts")


def _validate_append_order(appended: bytes) -> None:
    last_by_pair: dict[str, datetime] = {}
    for raw in appended.decode("utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("forecast append contains malformed JSON") from exc
        if not isinstance(item, dict):
            raise ValueError("forecast append contains a non-object row")
        pair = str(item.get("pair") or "")
        timestamp = replay._parse_time(item.get("timestamp_utc"))
        if not pair or timestamp is None:
            raise ValueError("forecast append contains missing pair/timestamp")
        previous = last_by_pair.get(pair)
        if previous is not None and timestamp < previous:
            raise ValueError("forecast append contains backdated rows")
        last_by_pair[pair] = timestamp


def _validate_emission_receipt_chain(payload: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_sha: str | None = None
    for raw in payload.decode("utf-8").splitlines():
        if not raw.strip():
            continue
        expected_sequence = len(rows) + 1
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("forecast emission receipt ledger is malformed") from exc
        if not isinstance(item, dict) or set(item) != EMISSION_RECEIPT_KEYS:
            raise ValueError("forecast emission receipt schema invalid")
        if item.get("schema_version") != "QR_FORECAST_EMISSION_RECEIPT_V1":
            raise ValueError("forecast emission receipt version invalid")
        if item.get("operation") not in {"APPEND", "REPLACE"}:
            raise ValueError("forecast emission receipt operation invalid")
        if item.get("sequence") != expected_sequence:
            raise ValueError("forecast emission receipt sequence invalid")
        if item.get("previous_receipt_sha256") != previous_sha:
            raise ValueError("forecast emission receipt chain broken")
        body = {key: value for key, value in item.items() if key != "receipt_sha256"}
        expected_sha = _content_sha256(body)
        if item.get("receipt_sha256") != expected_sha:
            raise ValueError("forecast emission receipt digest mismatch")
        if replay._parse_time(item.get("recorded_at_utc")) is None:
            raise ValueError("forecast emission receipt recorded time invalid")
        if replay._parse_time(item.get("forecast_timestamp_utc")) is None:
            raise ValueError("forecast emission receipt forecast time invalid")
        if not str(item.get("pair") or "") or not str(item.get("cycle_id") or ""):
            raise ValueError("forecast emission receipt identity invalid")
        previous_sha = str(item["receipt_sha256"])
        rows.append(item)
    return rows


def _forecast_receipt_blockers(
    *,
    forecast_path: Path,
    selected: Sequence[Any],
    receipts: Sequence[Mapping[str, Any]],
    pair: str,
    forecast_from: datetime,
    forecast_to: datetime,
) -> list[str]:
    blockers: list[str] = []
    latest_receipt_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for receipt in receipts:
        receipt_pair = str(receipt.get("pair") or "")
        forecast_timestamp = _required_time(
            receipt.get("forecast_timestamp_utc"),
            "receipt.forecast_timestamp_utc",
        )
        if receipt_pair != pair or not forecast_from <= forecast_timestamp < forecast_to:
            continue
        recorded_at = _required_time(receipt.get("recorded_at_utc"), "receipt.recorded_at_utc")
        if recorded_at < forecast_timestamp - timedelta(minutes=1):
            blockers.append("FORECAST_EMISSION_TIME_INCONSISTENT")
        if recorded_at > forecast_timestamp + timedelta(minutes=15):
            blockers.append("LATE_FORECAST_EMISSION_OR_REPLACEMENT")
        key = (str(receipt.get("cycle_id") or ""), receipt_pair)
        latest_receipt_by_key[key] = receipt

    raw_lines = forecast_path.read_text(encoding="utf-8").splitlines()
    for row in selected:
        cycle_id = str(row.cycle_id or "")
        if not cycle_id:
            blockers.append("FORECAST_EMISSION_RECEIPT_CYCLE_ID_MISSING")
            continue
        if row.source_index < 0 or row.source_index >= len(raw_lines):
            blockers.append("FORECAST_EMISSION_RECEIPT_SOURCE_INDEX_INVALID")
            continue
        try:
            raw_payload = json.loads(raw_lines[row.source_index])
        except json.JSONDecodeError:
            blockers.append("FORECAST_EMISSION_RECEIPT_ROW_INVALID")
            continue
        if not isinstance(raw_payload, dict):
            blockers.append("FORECAST_EMISSION_RECEIPT_ROW_INVALID")
            continue
        row_sha = _content_sha256(raw_payload)
        receipt = latest_receipt_by_key.get((cycle_id, row.pair))
        if receipt is None:
            blockers.append("FORECAST_EMISSION_RECEIPT_MISSING")
        elif receipt.get("forecast_row_sha256") != row_sha:
            blockers.append("FORECAST_EMISSION_RECEIPT_ROW_MISMATCH")
    return sorted(set(blockers))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    lock = sub.add_parser("lock", help="freeze one future candidate before its cohort starts")
    lock.add_argument("--candidate-contract", type=Path, required=True)
    lock.add_argument("--training-report", type=Path, required=True)
    lock.add_argument("--forecast-history", type=Path, required=True)
    lock.add_argument("--holdout-from", required=True)
    lock.add_argument("--holdout-to", required=True)
    lock.add_argument("--granularity", default="S5")
    lock.add_argument("--truth-root", type=Path, action="append", required=True)
    lock.add_argument("--output-dir", type=Path, default=Path("logs/reports/forecast_forward_holdout"))
    evaluate = sub.add_parser("evaluate", help="score only the immutable locked candidate")
    evaluate.add_argument("--lock", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, default=Path("logs/reports/forecast_forward_holdout"))
    return parser.parse_args()


def _context_label(context: Mapping[str, Any], *path: str) -> str:
    current: object = context
    for key in path:
        if not isinstance(current, Mapping):
            return "MISSING"
        current = current.get(key)
    return str(current or "MISSING").strip().upper()


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(f"{label} fields invalid: missing={sorted(expected - actual)} unknown={sorted(actual - expected)}")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _finite(value: object, label: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _typed_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a JSON number")
    return _finite(value, label)


def _optional_finite(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if not math.isnan(number) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "INFINITY" if value > 0 else "-INFINITY"
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _required_time(value: object, label: str) -> datetime:
    parsed = replay._parse_time(value)
    if parsed is None:
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp")
    return parsed


def _granularity_delta(value: str) -> timedelta:
    delta = replay._granularity_delta(str(value).upper())
    if delta is None:
        raise ValueError("unsupported granularity")
    return delta


def _content_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _semantic_dependency_sha256s() -> dict[str, str]:
    repo_root = SCRIPT_ROOT.parent
    paths = (
        repo_root / "src/quant_rabbit/broker/oanda.py",
        repo_root / "src/quant_rabbit/instruments.py",
        repo_root / "src/quant_rabbit/paths.py",
        repo_root / "src/quant_rabbit/strategy/forecast_technical_context.py",
    )
    return {
        str(path.resolve()): _sha256_bytes(path.read_bytes())
        for path in paths
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_pretty(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


if __name__ == "__main__":
    raise SystemExit(main())
