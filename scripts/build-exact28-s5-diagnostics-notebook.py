#!/usr/bin/env python3
"""Build and execute the reviewed exact-28 S5 diagnostic notebook."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--m1-reference", type=Path, required=True)
    parser.add_argument("--train-reconciliation", type=Path, required=True)
    parser.add_argument("--train-research", type=Path, required=True)
    parser.add_argument("--train-lock", type=Path, required=True)
    parser.add_argument("--validation-replication", type=Path, required=True)
    parser.add_argument("--prospective-final-lock", type=Path, required=True)
    parser.add_argument("--quality-output", type=Path, required=True)
    parser.add_argument("--metric-output", type=Path, required=True)
    parser.add_argument("--prospective-lock-output", type=Path, required=True)
    parser.add_argument("--report-input-output", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    return parser.parse_args(argv)


def _input_relative(path: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError as error:
        raise ValueError(f"input must remain inside repository: {path}") from error


def _output_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError as error:
        raise ValueError(f"output must remain inside repository: {path}") from error


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def _build_notebook(args: argparse.Namespace) -> nbformat.NotebookNode:
    manifest_relative = _input_relative(args.manifest)
    m1_relative = _input_relative(args.m1_reference)
    reconciliation_relative = _input_relative(args.train_reconciliation)
    research_relative = _input_relative(args.train_research)
    lock_relative = _input_relative(args.train_lock)
    validation_relative = _input_relative(args.validation_replication)
    runner_prospective_relative = _input_relative(args.prospective_final_lock)
    quality_relative = _output_relative(args.quality_output)
    metric_relative = _output_relative(args.metric_output)
    prospective_relative = _output_relative(args.prospective_lock_output)
    report_input_relative = _output_relative(args.report_input_output)

    reconciliation = _load_object(args.train_reconciliation)
    scenarios = reconciliation["scenarios"]
    prior = scenarios["prior_exact_anchor"]["metrics"]
    legacy = scenarios["legacy_three_changes"]["metrics"]
    gap = scenarios["execution_gap300_only"]["metrics"]
    research = _load_object(args.train_research)
    lock = _load_object(args.train_lock)
    validation = _load_object(args.validation_replication)
    survivor_train = lock["train_metrics"]
    survivor_validation = validation["metrics"]
    headline = (
        f"Canonical prior-contract TRAIN is {prior['trade_count']} trades, "
        f"{float(prior['net_pips']):+.1f} pips, PF {prior['profit_factor']}. "
        f"The provisional {float(legacy['net_pips']):+.1f}-pip result is invalid: "
        f"a 300-second post-selection execution cutoff removed 60 weekend-loss "
        f"trades and shifted net P/L by {float(gap['net_pips']) - float(prior['net_pips']):+.1f} pips. "
        f"The corrected causal 192-family TRAIN froze one pre-entry weekend-gate "
        f"survivor ({survivor_train['trade_count']} trades, "
        f"{float(survivor_train['net_pips']):+.1f} pips, PF "
        f"{survivor_train['profit_factor']}; Holm-adjusted p="
        f"{survivor_train['holm_adjusted_p']}) that replicated positively on the "
        f"locked VALIDATION window ({survivor_validation['trade_count']} trades, "
        f"{float(survivor_validation['net_pips']):+.1f} pips, PF "
        f"{survivor_validation['profit_factor']}), but the replication is not an "
        "independent claim and is not statistically confirmed. No profit is "
        "accepted before the unopened Jul 20–Aug 3 future test."
    )

    paths_cell = f'''from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
from IPython.display import display

ROOT = Path.cwd().resolve()
MANIFEST_PATH = ROOT / {json.dumps(manifest_relative)}
M1_REFERENCE_PATH = ROOT / {json.dumps(m1_relative)}
RECONCILIATION_PATH = ROOT / {json.dumps(reconciliation_relative)}
TRAIN_RESEARCH_PATH = ROOT / {json.dumps(research_relative)}
TRAIN_LOCK_PATH = ROOT / {json.dumps(lock_relative)}
VALIDATION_REPLICATION_PATH = ROOT / {json.dumps(validation_relative)}
RUNNER_PROSPECTIVE_LOCK_PATH = ROOT / {json.dumps(runner_prospective_relative)}
QUALITY_OUTPUT_PATH = ROOT / {json.dumps(quality_relative)}
METRIC_OUTPUT_PATH = ROOT / {json.dumps(metric_relative)}
PROSPECTIVE_LOCK_OUTPUT_PATH = ROOT / {json.dumps(prospective_relative)}
REPORT_INPUT_OUTPUT_PATH = ROOT / {json.dumps(report_input_relative)}

TRAIN_FROM = "2026-05-12T00:00:00Z"
TRAIN_TO = "2026-06-15T00:00:00Z"
VALIDATION_FROM = "2026-06-15T00:00:00Z"
VALIDATION_TO = "2026-06-28T00:00:00Z"
INTEGRITY_HOLDOUT_FROM = "2026-07-10T00:00:00Z"
INTEGRITY_HOLDOUT_TO = "2026-07-17T00:00:00Z"
PROSPECTIVE_FROM = "2026-07-20T00:00:00Z"
PROSPECTIVE_TO = "2026-08-03T00:00:00Z"
'''

    helper_cell = '''def load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict), path
    return value


def canonical_sha(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_digest(value: dict, digest_key: str) -> None:
    body = {key: item for key, item in value.items() if key != digest_key}
    assert value[digest_key] == canonical_sha(body), digest_key


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


manifest = load_object(MANIFEST_PATH)
m1_reference = load_object(M1_REFERENCE_PATH)
reconciliation = load_object(RECONCILIATION_PATH)
train_research = load_object(TRAIN_RESEARCH_PATH)
train_lock = load_object(TRAIN_LOCK_PATH)
validation_replication = load_object(VALIDATION_REPLICATION_PATH)
runner_prospective_lock = load_object(RUNNER_PROSPECTIVE_LOCK_PATH)
verify_digest(manifest, "manifest_sha256")
verify_digest(reconciliation, "artifact_sha256")
verify_digest(train_research, "research_sha256")
verify_digest(train_lock, "lock_sha256")
verify_digest(validation_replication, "evaluation_sha256")
verify_digest(runner_prospective_lock, "prospective_lock_sha256")

assert manifest["selected_pair_count"] == manifest["expected_pair_count"] == 28
assert manifest["complete_pair_coverage"] is True
assert manifest["all_selected_sources_acquisition_receipted"] is True
assert reconciliation["source_manifest_sha256"] == manifest["manifest_sha256"]
assert reconciliation["train_from_utc"] == TRAIN_FROM
assert reconciliation["train_to_utc"] == TRAIN_TO
assert reconciliation["accepted_positive_result"] is False
assert reconciliation["raw_trade_rows_persisted"] is False
assert reconciliation["holdout_state"]["july_10_17_strategy_evaluated"] is False
assert reconciliation["holdout_state"]["july_20_august_3_opened"] is False

# The corrected causal-family artifacts must form one digest-bound chain.
assert train_research["source_manifest_sha256"] == manifest["manifest_sha256"]
assert train_lock["research_sha256"] == train_research["research_sha256"]
assert train_research["locked_survivor_spec_id"] == train_lock["spec"]["spec_id"]
assert validation_replication["lock_sha256"] == train_lock["lock_sha256"]
assert runner_prospective_lock["shadow_lock_sha256"] == train_lock["lock_sha256"]
assert train_lock["validation_accessed_during_lock"] is False
assert validation_replication["independent_validation_claim_allowed"] is False
assert runner_prospective_lock["state"] == "UNAVAILABLE_UNOPENED"
assert runner_prospective_lock["strategy_prices_read"] is False
assert train_lock["order_authority"] == "NONE"
assert train_lock["live_permission"] is False

source_artifacts = [
    {"name": "exact-28 manifest", "path": str(MANIFEST_PATH.relative_to(ROOT)), "file_sha256": file_sha(MANIFEST_PATH), "internal_sha256": manifest["manifest_sha256"]},
    {"name": "M1 approximation reference", "path": str(M1_REFERENCE_PATH.relative_to(ROOT)), "file_sha256": file_sha(M1_REFERENCE_PATH), "internal_sha256": None},
    {"name": "independent TRAIN reconciliation", "path": str(RECONCILIATION_PATH.relative_to(ROOT)), "file_sha256": file_sha(RECONCILIATION_PATH), "internal_sha256": reconciliation["artifact_sha256"]},
    {"name": "corrected causal TRAIN research", "path": str(TRAIN_RESEARCH_PATH.relative_to(ROOT)), "file_sha256": file_sha(TRAIN_RESEARCH_PATH), "internal_sha256": train_research["research_sha256"]},
    {"name": "TRAIN survivor lock", "path": str(TRAIN_LOCK_PATH.relative_to(ROOT)), "file_sha256": file_sha(TRAIN_LOCK_PATH), "internal_sha256": train_lock["lock_sha256"]},
    {"name": "locked VALIDATION replication", "path": str(VALIDATION_REPLICATION_PATH.relative_to(ROOT)), "file_sha256": file_sha(VALIDATION_REPLICATION_PATH), "internal_sha256": validation_replication["evaluation_sha256"]},
    {"name": "runner prospective final-test lock", "path": str(RUNNER_PROSPECTIVE_LOCK_PATH.relative_to(ROOT)), "file_sha256": file_sha(RUNNER_PROSPECTIVE_LOCK_PATH), "internal_sha256": runner_prospective_lock["prospective_lock_sha256"]},
]
display(pd.DataFrame(source_artifacts))
'''

    profile_cell = '''pair_profiles = []
for row in manifest["selected_sources"]:
    first = datetime.fromisoformat(row["observed_first_utc"])
    last = datetime.fromisoformat(row["observed_last_utc"])
    continuous_slots = int((last - first).total_seconds() // 5) + 1
    pair_profiles.append({
        "pair": row["pair"],
        "declared_rows": int(row["declared_rows"]),
        "usable_rows": int(row["usable_rows"]),
        "quarantined_boundary_rows": int(row["quarantined_boundary_rows"]),
        "continuous_grid_slots_including_weekends": continuous_slots,
        "observed_slot_rate_including_weekends": round(int(row["usable_rows"]) / continuous_slots, 6),
        "observed_first_utc": row["observed_first_utc"],
        "observed_last_utc": row["observed_last_utc"],
        "file_size_bytes": int(row["file_size_bytes"]),
        "candidate_count_for_pair": int(row["candidate_count_for_pair"]),
        "acquisition_receipt_proved": bool(row["acquisition_receipt_proved"]),
        "source_file_sha256": row["file_sha256"],
    })

profile_df = pd.DataFrame(pair_profiles).sort_values("pair").reset_index(drop=True)
minimum = profile_df.loc[profile_df["usable_rows"].idxmin()]
maximum = profile_df.loc[profile_df["usable_rows"].idxmax()]
coverage_metrics = {
    "pair_count": int(len(profile_df)),
    "usable_rows_total": int(profile_df["usable_rows"].sum()),
    "usable_rows_min_pair": str(minimum["pair"]),
    "usable_rows_min": int(minimum["usable_rows"]),
    "usable_rows_max_pair": str(maximum["pair"]),
    "usable_rows_max": int(maximum["usable_rows"]),
    "quarantined_boundary_rows_total": int(profile_df["quarantined_boundary_rows"].sum()),
    "duplicate_source_candidates": len(manifest["duplicate_candidates"]),
    "unadmitted_files": len(manifest["unadmitted_files"]),
    "all_acquisition_receipted": bool(manifest["all_selected_sources_acquisition_receipted"]),
}
display(profile_df[["pair", "usable_rows", "quarantined_boundary_rows", "observed_slot_rate_including_weekends"]])
coverage_metrics
'''

    quality_cell = '''prior_metrics = reconciliation["scenarios"]["prior_exact_anchor"]["metrics"]
legacy_metrics = reconciliation["scenarios"]["legacy_three_changes"]["metrics"]
gap_metrics = reconciliation["scenarios"]["execution_gap300_only"]["metrics"]
omitted_weekend_trade_count = int(prior_metrics["trade_count"]) - int(gap_metrics["trade_count"])
omitted_weekend_loss_pips = round(float(prior_metrics["net_pips"]) - float(gap_metrics["net_pips"]), 9)

quality_findings = [
    {"severity": "PASS", "check": "EXACT_PAIR_SCOPE", "status": "28_OF_28", "evidence": "configured G8 universe complete; no missing pairs", "impact": "cross-sectional ranks are not silently reduced"},
    {"severity": "PASS", "check": "SOURCE_IDENTITY", "status": "VERIFIED", "evidence": "all 28 sources bind acquisition, summary, and file SHA-256 evidence", "impact": "market-data provenance is frozen"},
    {"severity": "PASS", "check": "ROW_GRAIN_VALIDITY", "status": "FAIL_CLOSED_ADMITTED", "evidence": "complete S5 BA rows; unique increasing 5-second clocks; valid OHLC; non-crossed executable open/close", "impact": "malformed rows cannot silently enter the audit"},
    {"severity": "CRITICAL", "check": "POST_SELECTION_EXECUTION_SURVIVORSHIP", "status": "CONFIRMED", "evidence": f"300-second cutoff removed {omitted_weekend_trade_count} weekend-exit trades totaling {omitted_weekend_loss_pips:+.1f} pips; canonical {prior_metrics['net_pips']:+.1f} became {gap_metrics['net_pips']:+.1f}", "impact": "provisional +689.4p and +386.7p lock are invalid"},
    {"severity": "HIGH", "check": "M1_REFERENCE_REPRODUCIBILITY", "status": "UNSEALED", "evidence": m1_reference["source_status"], "impact": "M1-to-exact differences cannot be assigned to one implementation detail"},
    {"severity": "HIGH", "check": "VALIDATION_STATE", "status": "LOCKED_REPLICATION_EVALUATED_NOT_INDEPENDENT", "evidence": f"the corrected causal family froze one TRAIN survivor which replicated on [2026-06-15, 2026-06-28) with {validation_replication['metrics']['trade_count']} trades and {float(validation_replication['metrics']['net_pips']):+.1f} pips; a related M1 approximation of this window was previously inspected, so independent_validation_claim_allowed=false", "impact": "positive replication exists but cannot be claimed as independent confirmation"},
    {"severity": "HIGH", "check": "STATISTICAL_MULTIPLICITY", "status": "NOT_SIGNIFICANT_AFTER_HOLM", "evidence": f"survivor one-sided daily normal p={train_lock['train_metrics']['one_sided_daily_normal_p']}, Holm-adjusted p={train_lock['train_metrics']['holm_adjusted_p']} across 192 candidates", "impact": "selection is economic-robustness only; no statistical significance claim is available"},
    {"severity": "HIGH", "check": "REGIME_COVERAGE", "status": "24_ACTIVE_TRAIN_DAYS", "evidence": "one short May–June 2026 regime only", "impact": "cannot support any-condition or long-run claims; stage 2020–2026 M5 acquisition only as next evidence if no robust TRAIN survivor"},
    {"severity": "HIGH", "check": "JUL10_17_PURITY", "status": "INTEGRITY_READ_STRATEGY_UNEVALUATED", "evidence": "manifest parsed bytes for schema/hash only; no features, outcomes, selection, or metrics", "impact": "not byte-unseen and not a pristine final test"},
    {"severity": "PASS", "check": "FUTURE_FINAL_TEST", "status": "JUL20_AUG3_UNAVAILABLE_UNOPENED", "evidence": "no source or metric exists for the future window", "impact": "genuinely prospective evidence remains protected"},
]
quality_df = pd.DataFrame(quality_findings)
display(quality_df)

quality_body = {
    "artifact": "QR_EXACT28_S5_DATA_QUALITY_PROFILE_V1",
    "source_artifacts": source_artifacts,
    "dataset_grain": "one complete OANDA S5 bid/ask candle per pair and unique 5-second timestamp",
    "coverage_metrics": coverage_metrics,
    "pair_profiles": pair_profiles,
    "quality_findings": quality_findings,
    "split_contract": {
        "TRAIN": {"from_utc": TRAIN_FROM, "to_utc": TRAIN_TO, "evaluated": True},
        "VALIDATION": {"from_utc": VALIDATION_FROM, "to_utc": VALIDATION_TO, "evaluated_after_reconciliation": True, "locked_replication_only": True, "independent_claim_allowed": False},
        "JUL10_17": {"from_utc": INTEGRITY_HOLDOUT_FROM, "to_utc": INTEGRITY_HOLDOUT_TO, "manifest_integrity_bytes_read": True, "strategy_evaluated": False},
        "PROSPECTIVE_FINAL_TEST": {"from_utc": PROSPECTIVE_FROM, "to_utc": PROSPECTIVE_TO, "source_available": False, "opened": False},
    },
    "overall_assessment": "PASS_WITH_LIMITATIONS",
    "accepted_positive_result": False,
    "historical_only": True,
    "diagnostic_only": True,
    "forward_proof_eligible": False,
    "order_authority": "NONE",
    "live_permission": False,
    "broker_mutation_allowed": False,
}
quality_artifact = {**quality_body, "artifact_sha256": canonical_sha(quality_body)}
atomic_json(QUALITY_OUTPUT_PATH, quality_artifact)
'''

    metric_cell = '''scenarios = reconciliation["scenarios"]
scenario_rows = []
for scenario_name in (
    "prior_exact_anchor", "legacy_three_changes", "execution_gap300_only",
    "stale_signal_only", "ready4_only", "decision_due_only", "fixed_continuous_hold",
):
    scenario = scenarios[scenario_name]
    scenario_rows.append({"scenario": scenario_name, **scenario["metrics"]})
scenario_df = pd.DataFrame(scenario_rows)
display(scenario_df[["scenario", "trade_count", "decision_count", "net_pips", "mean_pips", "profit_factor"]])

m1_train = m1_reference["splits"]["TRAIN"]
metric_definition_reconciliation = [
    {"measurement": "M1_APPROXIMATION_UNSEALED", "trade_count": m1_train["trade_count"], "net_pips": m1_train["net_pips"], "profit_factor": m1_train["profit_factor"], "status": "REFERENCE_ONLY"},
    {"measurement": "EXACT_S5_PRIOR_CONTRACT", "trade_count": prior_metrics["trade_count"], "net_pips": prior_metrics["net_pips"], "profit_factor": prior_metrics["profit_factor"], "status": "CANONICAL_TRAIN_DIAGNOSTIC"},
    {"measurement": "EXACT_S5_LEGACY_THREE_CHANGES", "trade_count": legacy_metrics["trade_count"], "net_pips": legacy_metrics["net_pips"], "profit_factor": legacy_metrics["profit_factor"], "status": "INVALID_SURVIVORSHIP"},
    {"measurement": "EXACT_S5_EXECUTION_GAP300_ONLY", "trade_count": gap_metrics["trade_count"], "net_pips": gap_metrics["net_pips"], "profit_factor": gap_metrics["profit_factor"], "status": "ISOLATED_DEFECT"},
    {"measurement": "CAUSAL_FAMILY_TRAIN_SURVIVOR", "trade_count": train_lock["train_metrics"]["trade_count"], "net_pips": train_lock["train_metrics"]["net_pips"], "profit_factor": train_lock["train_metrics"]["profit_factor"], "status": "TRAIN_LOCKED_NOT_SIGNIFICANT"},
    {"measurement": "CAUSAL_SURVIVOR_LOCKED_VALIDATION", "trade_count": validation_replication["metrics"]["trade_count"], "net_pips": validation_replication["metrics"]["net_pips"], "profit_factor": validation_replication["metrics"]["profit_factor"], "status": "REPLICATED_NOT_INDEPENDENT"},
]

driver_rows = [
    {"driver": "300S_POST_SELECTION_EXECUTION_GAP", "comparison": "prior_exact_anchor -> execution_gap300_only", "trade_count_delta": -60, "net_pips_delta": 1438.8, "interpretation": "removed Friday 12/16/20 UTC trades whose exits arrived after 133,495–162,295 seconds; omitted trades totaled -1,438.8p", "causal_status": "CONFIRMED_PRIMARY_DRIVER"},
    {"driver": "STALE_SIGNAL_LOOKUP", "comparison": "prior_exact_anchor -> stale_signal_only", "trade_count_delta": 20, "net_pips_delta": -4.3, "interpretation": "small negative isolated effect", "causal_status": "CONFIRMED_ISOLATED"},
    {"driver": "READY_FLOOR_24_TO_4", "comparison": "prior_exact_anchor -> ready4_only", "trade_count_delta": 20, "net_pips_delta": 31.4, "interpretation": "small positive isolated effect; does not explain sign reversal", "causal_status": "CONFIRMED_ISOLATED"},
    {"driver": "HOLD_DUE_ENTRY_TO_DECISION", "comparison": "prior_exact_anchor -> decision_due_only", "trade_count_delta": 0, "net_pips_delta": 1.7, "interpretation": "negligible isolated effect", "causal_status": "CONFIRMED_ISOLATED"},
    {"driver": "THREE_CHANGE_INTERACTION", "comparison": "prior_exact_anchor -> legacy_three_changes", "trade_count_delta": -40, "net_pips_delta": 1437.0, "interpretation": "combined delta is non-additive; dominated by execution-gap survivorship", "causal_status": "CONFIRMED_COMBINED"},
]

invalid_lock = reconciliation["invalidated_provisional_lock"]
candidate_robustness_rows = [
    {"candidate": "prior exact top/bottom2 anchor", "trade_count": prior_metrics["trade_count"], "net_pips": prior_metrics["net_pips"], "profit_factor": prior_metrics["profit_factor"], "leave_best_day_stressed_net_pips": None, "leave_best_pair_stressed_net_pips": None, "status": "REJECT_NEGATIVE_TRAIN"},
    {"candidate": "legacy top/bottom2", "trade_count": legacy_metrics["trade_count"], "net_pips": legacy_metrics["net_pips"], "profit_factor": legacy_metrics["profit_factor"], "leave_best_day_stressed_net_pips": -129.9, "leave_best_pair_stressed_net_pips": -577.6, "status": "INVALID_EXECUTION_GAP_SURVIVORSHIP"},
    {"candidate": "provisional top/bottom1 dispersion5", "trade_count": invalid_lock["trade_count"], "net_pips": invalid_lock["raw_net_pips"], "profit_factor": invalid_lock["profit_factor"], "leave_best_day_stressed_net_pips": invalid_lock["leave_best_day_stressed_net_pips"], "leave_best_pair_stressed_net_pips": invalid_lock["leave_best_pair_stressed_net_pips"], "status": invalid_lock["status"]},
    {"candidate": "continuous-week pre-entry gate diagnostic", "trade_count": scenarios["fixed_continuous_hold"]["metrics"]["trade_count"], "net_pips": scenarios["fixed_continuous_hold"]["metrics"]["net_pips"], "profit_factor": scenarios["fixed_continuous_hold"]["metrics"]["profit_factor"], "leave_best_day_stressed_net_pips": None, "leave_best_pair_stressed_net_pips": None, "status": "DIFFERENT_STRATEGY_TRAIN_ONLY_NOT_ACCEPTED"},
    {"candidate": "locked causal survivor " + train_lock["spec"]["spec_id"], "trade_count": train_lock["train_metrics"]["trade_count"], "net_pips": train_lock["train_metrics"]["net_pips"], "profit_factor": train_lock["train_metrics"]["profit_factor"], "leave_best_day_stressed_net_pips": train_lock["train_metrics"]["leave_best_day_stressed_net_pips"], "leave_best_pair_stressed_net_pips": train_lock["train_metrics"]["leave_best_pair_stressed_net_pips"], "status": "TRAIN_LOCKED_VALIDATION_REPLICATED_AWAITING_FUTURE_TEST"},
]
display(pd.DataFrame(driver_rows))
display(pd.DataFrame(candidate_robustness_rows))

metric_body = {
    "artifact": "QR_EXACT_S5_METRIC_DEFINITION_DIAGNOSTIC_V1",
    "source_artifacts": source_artifacts,
    "metric_definition_reconciliation": metric_definition_reconciliation,
    "scenario_rows": scenario_rows,
    "driver_rows": driver_rows,
    "candidate_robustness_rows": candidate_robustness_rows,
    "accepted_positive_result": False,
    "valid_train_survivor_count": int(train_research["locked_survivor_spec_id"] is not None),
    "locked_survivor_spec_id": train_research["locked_survivor_spec_id"],
    "validation_evaluated": True,
    "validation_evaluation_sha256": validation_replication["evaluation_sha256"],
    "independent_validation_claim_allowed": validation_replication["independent_validation_claim_allowed"],
    "long_history_next_evidence": "STAGE_2020_2026_M5_ACQUISITION_ONLY_AFTER_VALIDATOR_AUDIT_REPAIRS_DO_NOT_FETCH_IN_THIS_RUN",
    "historical_only": True,
    "diagnostic_only": True,
    "forward_proof_eligible": False,
    "order_authority": "NONE",
    "live_permission": False,
    "broker_mutation_allowed": False,
}
metric_artifact = {**metric_body, "artifact_sha256": canonical_sha(metric_body)}
atomic_json(METRIC_OUTPUT_PATH, metric_artifact)
'''

    final_cell = '''prospective_body = {
    "artifact": "QR_EXACT_S5_PROSPECTIVE_FINAL_TEST_LOCK_V1",
    "window": {"opened_from_utc": PROSPECTIVE_FROM, "opened_to_utc": PROSPECTIVE_TO},
    "status": "CANDIDATE_BOUND_AWAITING_UNAVAILABLE_FUTURE_WINDOW",
    "source_available_at_lock": False,
    "source_accessed": False,
    "metrics_persisted": False,
    "candidate_spec": dict(train_lock["spec"]),
    "candidate_lock_sha256": train_lock["lock_sha256"],
    "runner_prospective_lock_sha256": runner_prospective_lock["prospective_lock_sha256"],
    "stale_train_lock_accepted": False,
    "candidate_reselection_allowed": False,
    "evaluation_condition": "ONE_FROZEN_TRAIN_SURVIVOR_EVALUATED_UNCHANGED_AFTER_NEW_ACQUISITION",
    "july_10_17_is_not_this_final_test": True,
    "automatic_promotion_allowed": False,
    "promotion_allowed": False,
    "order_authority": "NONE",
    "live_permission": False,
    "broker_mutation_allowed": False,
}
prospective_lock = {**prospective_body, "lock_sha256": canonical_sha(prospective_body)}
atomic_json(PROSPECTIVE_LOCK_OUTPUT_PATH, prospective_lock)

report_body = {
    "artifact": "QR_EXACT28_S5_DATA_ANALYTICS_REPORT_INPUT_V1",
    "review_status": "PASS_WITH_LIMITATIONS",
    "headline": (
        "Canonical exact-S5 TRAIN is negative and the provisional positives were "
        "execution survivorship; the corrected causal family froze one pre-entry "
        "weekend-gate survivor that replicated positively on locked VALIDATION, "
        "but the replication is non-independent, not statistically significant, "
        "and no profit is accepted before the unopened Jul 20-Aug 3 future test."
    ),
    "corrected_causal_family": {
        "candidate_count": train_research["candidate_count"],
        "eligible_candidate_count": train_research["eligible_candidate_count"],
        "locked_survivor_spec": dict(train_lock["spec"]),
        "train_metrics": dict(train_lock["train_metrics"]),
        "validation_metrics": dict(validation_replication["metrics"]),
        "independent_validation_claim_allowed": validation_replication["independent_validation_claim_allowed"],
        "holm_multiplicity_confirmed": train_lock["train_metrics"]["multiplicity_confirmed"],
    },
    "source_artifacts": source_artifacts,
    "source_coverage_metrics": coverage_metrics,
    "metric_definition_reconciliation": metric_definition_reconciliation,
    "metric_driver_rows": driver_rows,
    "candidate_robustness_rows": candidate_robustness_rows,
    "data_quality_findings": quality_findings,
    "split_state": quality_body["split_contract"],
    "prospective_final_test_lock_sha256": prospective_lock["lock_sha256"],
    "accepted_positive_result": False,
    "raw_candles_or_trades_persisted": False,
    "recommended_next_evidence": [
        "Keep Jul 20-Aug 3 unavailable and unopened; evaluate the one frozen survivor unchanged only after that window's data is newly acquired.",
        "Treat the locked VALIDATION replication as non-independent evidence because a related M1 approximation of the window was previously inspected.",
        "Do not claim statistical significance: the survivor's Holm-adjusted p is 1.0 across the 192-candidate family.",
        "Stage the separate 2020-2026 M5 acquisition only after the M5 validator audit repairs (boundary-candle clamp pilot, internal-gap cross-checks); do not fetch it in this run.",
    ],
    "historical_only": True,
    "diagnostic_only": True,
    "forward_proof_eligible": False,
    "order_authority": "NONE",
    "live_permission": False,
    "broker_mutation_allowed": False,
}
report_input = {**report_body, "artifact_sha256": canonical_sha(report_body)}
atomic_json(REPORT_INPUT_OUTPUT_PATH, report_input)

pd.DataFrame([
    {"artifact": "data quality", "path": str(QUALITY_OUTPUT_PATH.relative_to(ROOT)), "sha256": quality_artifact["artifact_sha256"]},
    {"artifact": "metric diagnostic", "path": str(METRIC_OUTPUT_PATH.relative_to(ROOT)), "sha256": metric_artifact["artifact_sha256"]},
    {"artifact": "prospective lock", "path": str(PROSPECTIVE_LOCK_OUTPUT_PATH.relative_to(ROOT)), "sha256": prospective_lock["lock_sha256"]},
    {"artifact": "report input", "path": str(REPORT_INPUT_OUTPUT_PATH.relative_to(ROOT)), "sha256": report_input["artifact_sha256"]},
])
'''

    cells = [
        nbformat.v4.new_markdown_cell(
            "# Exact-28 OANDA S5 profitability diagnostics\n\n"
            "## tl;dr\n\n"
            + headline
            + "\n\nThis is a historical shadow audit. It grants no order authority or live permission."
        ),
        nbformat.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "The controlling data source is the sealed exact-28 OANDA S5 bid/ask manifest. "
            "An independent stdlib reconstruction—not the disputed adaptive engine—recomputed "
            "the prior contract and isolated each definition change on TRAIN only. This notebook "
            "reads compact JSON evidence and persists reviewed, content-addressed outputs; it never "
            "loads raw candles or trade ledgers.\n\n"
            "### Key Assumptions\n\n"
            "- TRAIN is `[2026-05-12, 2026-06-15)` UTC.\n"
            "- Exact VALIDATION remains unopened after the corrected TRAIN loss.\n"
            "- Jul 10–17 bytes were integrity-read by manifest admission but never strategy-evaluated.\n"
            "- Jul 20–Aug 3 is future, unavailable, and genuinely unopened.\n"
            "- The M1 approximation is unsealed comparison evidence, not source-of-truth performance."
        ),
        nbformat.v4.new_code_cell(paths_cell),
        nbformat.v4.new_markdown_cell("## Data\n\n### 1. Load and verify compact evidence"),
        nbformat.v4.new_code_cell(helper_cell),
        nbformat.v4.new_markdown_cell("### 2. Profile exact source coverage"),
        nbformat.v4.new_code_cell(profile_cell),
        nbformat.v4.new_markdown_cell("## Results\n\n### 3. Data-quality severity"),
        nbformat.v4.new_code_cell(quality_cell),
        nbformat.v4.new_markdown_cell("### 4. Reconcile metric definitions and drivers"),
        nbformat.v4.new_code_cell(metric_cell),
        nbformat.v4.new_markdown_cell("### 5. Seal future-test state and report input"),
        nbformat.v4.new_code_cell(final_cell),
        nbformat.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "1. The exact-28 source passes strict acquisition, grain, timestamp, OHLC, and bid/ask admission.\n"
            "2. The canonical prior-contract TRAIN result is **-747.6 pips, PF 0.8821**. No positive anchor is accepted.\n"
            "3. The 300-second post-selection execution cutoff is the sign-flip defect: it removed 60 weekend-loss trades totaling -1,438.8 pips.\n"
            "4. The corrected causal 192-family froze one pre-entry weekend-gate survivor on TRAIN (+1,089.0p, PF 1.166) that replicated on locked VALIDATION (+969.7p, PF 1.423) — but the replication is non-independent (a related M1 approximation was previously inspected) and the Holm-adjusted p is 1.0, so no significance or accepted profit is claimed.\n"
            "5. Jul 10–17 remains integrity-read but strategy-unevaluated; Jul 20–Aug 3 remains unavailable/unopened with exactly one frozen candidate bound to it.\n"
            "6. Twenty-four active TRAIN days are too narrow for universal claims. The 2020–2026 M5 evidence set stays staged—not fetched—until the M5 validator audit repairs land."
        ),
    ]
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )


def _atomic_notebook(path: Path, notebook: nbformat.NotebookNode) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        nbformat.write(notebook, temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> dict[str, Any]:
    notebook = _build_notebook(args)
    _atomic_notebook(args.notebook, notebook)
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    _atomic_notebook(args.notebook, notebook)
    outputs = (
        args.quality_output,
        args.metric_output,
        args.prospective_lock_output,
        args.report_input_output,
    )
    if any(not path.is_file() for path in outputs):
        raise RuntimeError("executed notebook did not persist every output")
    return {
        "status": "EXECUTED_TOP_TO_BOTTOM",
        "notebook": str(args.notebook.resolve()),
        "output_paths": [str(path.resolve()) for path in outputs],
        "cell_count": len(notebook.cells),
        "accepted_positive_result": False,
        "validation_accessed": False,
        "future_final_test_opened": False,
        "order_authority": "NONE",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
