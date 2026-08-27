"""Sign-aware component independence and bounded achievability diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import component_worker_evidence_registry_v1 as v1


POLICY_PATH = "COMPONENT_WORKER_INDEPENDENCE_POLICY_V2.json"
OUTPUT_ROOT = "evidence/component_worker_registry_v2"
REGISTRY_PATH = f"{OUTPUT_ROOT}/component_worker_registry_v2.json"
DAILY_PORTFOLIO_PATH = f"{OUTPUT_ROOT}/fixed_equal_sleeve_daily_returns.jsonl"
V1_REGISTRY_PATH = v1.REGISTRY_PATH
ARMS = v1.ARMS
AUTHORITY = v1.AUTHORITY


class SignAwareEvidenceError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    return v1.sha256_file(path)


def embedded_hash(payload: dict[str, Any], field: str) -> str:
    return v1.embedded_hash(payload, field)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise SignAwareEvidenceError("correlation vectors differ")
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - left_mean) ** 2 for a in left)
                            * sum((b - right_mean) ** 2 for b in right))
    return numerator / denominator if denominator else 0.0


def cvar_loss(values: list[float], fraction: float) -> float:
    count = max(1, math.ceil(len(values) * fraction))
    return -statistics.fmean(sorted(values)[:count])


def quantile_floor(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def daily_maps(root: Path, candidate: dict) -> dict[str, dict[str, float]]:
    rows = load_jsonl(root / candidate["evidence"]["daily_worker_returns_path"])
    return {arm: {row["utc_day"]: row["returns"][arm] for row in rows} for arm in ARMS}


def exposure_map(root: Path, candidate: dict) -> dict[str, dict[str, float]]:
    rows = load_jsonl(root / candidate["evidence"]["currency_time_exposure_path"])
    return {row["timestamp"]: row["currency_exposure_nav"] for row in rows}


def currency_time_clusters(exposures: dict[str, dict[str, float]]) -> list[dict]:
    clusters = []
    prior_stamp = None
    prior_vector = None
    for stamp in sorted(exposures, key=v1.engine.frozen_v31.ns):
        vector = exposures[stamp]
        contiguous = prior_stamp is not None \
            and v1.engine.frozen_v31.ns(stamp) - v1.engine.frozen_v31.ns(prior_stamp) == 300_000_000_000
        if not contiguous or vector != prior_vector:
            clusters.append({"start": stamp, "currency_exposure_nav": vector})
        prior_stamp, prior_vector = stamp, vector
    return clusters


def signal_events(root: Path, candidate: dict) -> list[dict]:
    rows = load_jsonl(root / candidate["evidence"]["signal_ledger_path"])
    return [row for row in rows
            if "2026-05-01" <= row["fill_time"][:10] < "2026-07-01"
            and row["exit_time"][:10] < "2026-07-01"]


def pair_review(root: Path, left: dict, right: dict, policy: dict) -> dict:
    left_daily, right_daily = daily_maps(root, left), daily_maps(root, right)
    days = sorted(set(left_daily["EXECUTABLE_BASE"]) | set(right_daily["EXECUTABLE_BASE"]))
    left_base = [left_daily["EXECUTABLE_BASE"].get(day, 0.0) for day in days]
    right_base = [right_daily["EXECUTABLE_BASE"].get(day, 0.0) for day in days]
    rho = pearson(left_base, right_base)
    left_events, right_events = signal_events(root, left), signal_events(root, right)
    left_stamps = {row["fill_time"] for row in left_events}
    right_stamps = {row["fill_time"] for row in right_events}
    stamp_overlap = len(left_stamps & right_stamps) / len(left_stamps | right_stamps) \
        if left_stamps | right_stamps else 0.0
    left_pairs = {(row["fill_time"], row["pair"]): int(row["direction"]) for row in left_events}
    right_pairs = {(row["fill_time"], row["pair"]): int(row["direction"]) for row in right_events}
    shared_pairs = set(left_pairs) & set(right_pairs)
    pair_overlap = len(shared_pairs) / len(set(left_pairs) | set(right_pairs)) \
        if set(left_pairs) | set(right_pairs) else 0.0
    opposite_event_rate = sum(left_pairs[key] == -right_pairs[key] for key in shared_pairs) / len(shared_pairs) \
        if shared_pairs else 0.0

    left_exposure, right_exposure = exposure_map(root, left), exposure_map(root, right)
    common_stamps = set(left_exposure) & set(right_exposure)
    common_currency_observations = []
    for stamp in common_stamps:
        for currency in set(left_exposure[stamp]) & set(right_exposure[stamp]):
            left_value, right_value = left_exposure[stamp][currency], right_exposure[stamp][currency]
            if left_value and right_value:
                common_currency_observations.append(left_value * right_value < 0)
    exposure_sign_inversion = sum(common_currency_observations) / len(common_currency_observations) \
        if common_currency_observations else 0.0

    at_least_one_loss = sum(a < 0 or b < 0 for a, b in zip(left_base, right_base))
    both_loss = sum(a < 0 and b < 0 for a, b in zip(left_base, right_base))
    downside_co_loss = both_loss / at_least_one_loss if at_least_one_loss else 0.0
    tail_fraction = policy["complementarity_review"]["adverse_tail_fraction"]
    left_adverse = [left_daily["ADVERSE_STRESS"].get(day, 0.0) for day in days]
    right_adverse = [right_daily["ADVERSE_STRESS"].get(day, 0.0) for day in days]
    left_cut = quantile_floor(left_adverse, tail_fraction)
    right_cut = quantile_floor(right_adverse, tail_fraction)
    tail_indexes = [index for index, (a, b) in enumerate(zip(left_adverse, right_adverse))
                    if a <= left_cut or b <= right_cut]
    tail_left = [left_adverse[index] for index in tail_indexes]
    tail_right = [right_adverse[index] for index in tail_indexes]
    tail_correlation = pearson(tail_left, tail_right)
    tail_co_loss = sum(a < 0 and b < 0 for a, b in zip(tail_left, tail_right)) / len(tail_indexes)
    equal_adverse = [(a + b) / 2 for a, b in zip(left_adverse, right_adverse)]
    individual_cvar_mean = statistics.fmean([
        cvar_loss(left_adverse, tail_fraction), cvar_loss(right_adverse, tail_fraction),
    ])
    portfolio_cvar = cvar_loss(equal_adverse, tail_fraction)
    cvar_improvement = 1.0 - portfolio_cvar / individual_cvar_mean if individual_cvar_mean > 0 else 0.0
    portfolio_tail_count = max(1, math.ceil(len(equal_adverse) * tail_fraction))
    portfolio_tail_indexes = sorted(range(len(equal_adverse)), key=equal_adverse.__getitem__)[:portfolio_tail_count]
    left_cvar_contribution = -statistics.fmean(0.5 * left_adverse[index]
                                              for index in portfolio_tail_indexes)
    right_cvar_contribution = -statistics.fmean(0.5 * right_adverse[index]
                                               for index in portfolio_tail_indexes)

    left_clusters = currency_time_clusters(left_exposure)
    right_clusters = currency_time_clusters(right_exposure)
    combined_cluster_keys = {
        (item["start"], json.dumps(item["currency_exposure_nav"], sort_keys=True))
        for item in left_clusters + right_clusters
    }
    max_individual_clusters = max(len(left_clusters), len(right_clusters))
    cluster_gain = len(combined_cluster_keys) / max_individual_clusters - 1.0 \
        if max_individual_clusters else 0.0
    review = policy["complementarity_review"]
    same_structure = (left["family"], left["session"], left["horizon_seconds"]) \
        == (right["family"], right["session"], right["horizon_seconds"])
    mechanical_inversion = (
        pair_overlap >= review["mechanical_signal_inversion_minimum_pair_event_overlap"]
        and opposite_event_rate >= review["mechanical_signal_inversion_minimum_opposite_direction_rate"]
    )
    same_event_netting = pair_overlap > review["maximum_pair_event_jaccard_overlap"]
    complementarity_checks = {
        "signal_timestamp_overlap_pass": stamp_overlap <= review["maximum_signal_timestamp_jaccard_overlap"],
        "pair_event_overlap_pass": pair_overlap <= review["maximum_pair_event_jaccard_overlap"],
        "mechanical_signal_inversion_absent": not mechanical_inversion,
        "same_structure_duplicate_absent": not same_structure,
        "downside_co_loss_pass": downside_co_loss <= review["maximum_downside_co_loss_rate"],
        "adverse_tail_co_loss_pass": tail_co_loss <= review["maximum_adverse_tail_co_loss_rate"],
        "adverse_cvar_improvement_pass": cvar_improvement
        >= review["minimum_equal_sleeve_adverse_cvar_improvement_fraction"],
        "currency_time_cluster_n_eff_gain_pass": cluster_gain
        >= review["minimum_currency_time_cluster_n_eff_gain_fraction"],
        "same_event_netting_absent": not same_event_netting,
    }
    positive_limit = policy["correlation_routing"]["positive_duplicate_if_strictly_greater_than"]
    negative_limit = policy["correlation_routing"]["negative_complementarity_review_if_strictly_less_than"]
    if rho > positive_limit:
        route, classification, passed = (
            "POSITIVE_CORRELATION_DUPLICATE_REVIEW", "DUPLICATE_POSITIVE_CORRELATION", False,
        )
    elif rho < negative_limit:
        route = "COMPLEMENTARITY_REVIEW"
        passed = all(complementarity_checks.values())
        classification = "DIVERSIFYING_INDEPENDENT" if passed else "COMPLEMENTARITY_REVIEW_FAILED"
    else:
        route = "LOW_CORRELATION_INDEPENDENCE_REVIEW"
        passed = all(complementarity_checks.values())
        classification = "LOW_CORRELATION_INDEPENDENT" if passed else "LOW_CORRELATION_REVIEW_FAILED"
    return {
        "left_cycle": left["cycle_id"], "right_cycle": right["cycle_id"],
        "routing": route, "classification": classification, "independence_gate_passed": passed,
        "daily_base_return_correlation": rho,
        "signal_timestamp_jaccard_overlap": stamp_overlap,
        "pair_event_jaccard_overlap": pair_overlap,
        "shared_pair_events": len(shared_pairs),
        "opposite_direction_rate_on_shared_pair_events": opposite_event_rate,
        "currency_exposure_sign_inversion_rate_at_common_timestamps": exposure_sign_inversion,
        "same_family": left["family"] == right["family"],
        "same_session": left["session"] == right["session"],
        "same_horizon": left["horizon_seconds"] == right["horizon_seconds"],
        "daily_downside_co_loss_rate": downside_co_loss,
        "adverse_tail": {
            "fraction": tail_fraction, "observations": len(tail_indexes),
            "correlation": tail_correlation, "co_loss_rate": tail_co_loss,
            "individual_mean_cvar_loss": individual_cvar_mean,
            "fixed_equal_sleeve_cvar_loss": portfolio_cvar,
            "cvar_improvement_fraction": cvar_improvement,
            "fixed_equal_sleeve_cvar_contribution": {
                left["cycle_id"]: left_cvar_contribution,
                right["cycle_id"]: right_cvar_contribution,
                "sum": left_cvar_contribution + right_cvar_contribution,
            },
        },
        "currency_time_cluster_n_eff": {
            "left": len(left_clusters), "right": len(right_clusters),
            "combined": len(combined_cluster_keys), "gain_over_larger_individual_fraction": cluster_gain,
        },
        "complementarity_checks": complementarity_checks,
    }


def compound(values: list[float]) -> float:
    result = 1.0
    for value in values:
        result *= 1.0 + value
    return result


def achievability_diagnostic(root: Path, candidates: list[dict], policy: dict) -> tuple[dict, list[dict]]:
    maps = {candidate["cycle_id"]: daily_maps(root, candidate) for candidate in candidates}
    days = sorted(set().union(*(set(maps[item["cycle_id"]]["RAW_SIGNAL"]) for item in candidates)))
    rows = []
    for day in days:
        rows.append({
            "utc_day": day,
            "returns": {
                arm: statistics.fmean(maps[item["cycle_id"]][arm].get(day, 0.0) for item in candidates)
                for arm in ARMS
            },
            "allocation": "FIXED_EQUAL_COMPONENT_SLEEVES",
        })
    months = {"MONTH_2026_05": ("2026-05-01", "2026-06-01"),
              "MONTH_2026_06": ("2026-06-01", "2026-07-01")}
    month_results = {}
    for month, (start, end) in months.items():
        month_rows = [row for row in rows if start <= row["utc_day"] < end]
        month_results[month] = {}
        for arm in ("EXECUTABLE_BASE", "ADVERSE_STRESS"):
            multiple = compound([row["returns"][arm] for row in month_rows])
            candidate_logs = []
            for candidate in candidates:
                values = [maps[candidate["cycle_id"]][arm].get(day, 0.0)
                          for day in sorted({row["utc_day"] for row in month_rows})]
                candidate_logs.append(sum(math.log1p(value) for value in values))
            mean_worker_log = statistics.fmean(candidate_logs)
            required = math.ceil(math.log(2.0) / mean_worker_log) if mean_worker_log > 0 else None
            month_results[month][arm] = {
                "fixed_equal_sleeve_multiple": multiple,
                "cost_after_average_daily_log_growth": statistics.fmean(
                    math.log1p(row["returns"][arm]) for row in month_rows
                ),
                "gap_to_2x_multiple_points": 2.0 - multiple,
                "gap_to_2x_log_growth": math.log(2.0) - math.log(multiple),
                "linear_additive_uncapped_worker_lower_bound": required,
                "required_independent_worker_count_linear_diagnostic": required,
                "current_independent_worker_count": len(candidates),
                "linear_worker_count_is_diagnostic_only": True,
                "linear_worker_count_may_not_pass_profit_gate": True,
            }

    exposure_maps = [exposure_map(root, candidate) for candidate in candidates]
    max_currency_abs = 0.0
    for stamp in set().union(*(set(item) for item in exposure_maps)):
        combined: dict[str, float] = {}
        for exposures in exposure_maps:
            for currency, value in exposures.get(stamp, {}).items():
                combined[currency] = combined.get(currency, 0.0) + value / len(candidates)
        max_currency_abs = max([max_currency_abs] + [abs(value) for value in combined.values()])
    conservative_gross = 0.0
    for candidate in candidates:
        result = json.loads((root / candidate["evidence"]["result_path"]).read_text(encoding="utf-8"))
        conservative_gross += result["periods"]["WALK_FORWARD"]["EXECUTABLE_BASE"]["max_gross_exposure_nav"] \
            / len(candidates)
    diagnostic = {
        "observed_sealed_component_returns_only": True,
        "oracle_used": False,
        "allocation": "FIXED_EQUAL_COMPONENT_SLEEVES",
        "current_independent_worker_candidates": len(candidates),
        "monthly": month_results,
        "conservative_max_gross_exposure_nav": conservative_gross,
        "max_currency_abs_exposure_nav": max_currency_abs,
        "gross_cap": policy["portfolio_diagnostic"]["gross_leverage_cap"],
        "currency_cap": policy["portfolio_diagnostic"]["currency_abs_exposure_cap"],
        "gross_cap_pass": conservative_gross <= policy["portfolio_diagnostic"]["gross_leverage_cap"] + 1e-12,
        "currency_cap_pass": max_currency_abs <= policy["portfolio_diagnostic"]["currency_abs_exposure_cap"] + 1e-12,
        "terminal_liquidation_required": True,
        "post_hoc_leverage_used": False,
        "profit_gate_pass_inferred": False,
        "holdout_reproduction_inferred": False,
    }
    return diagnostic, rows


def binding_hashes(root: Path) -> dict[str, str]:
    paths = {
        "v41_result": "evidence/run_london_open_false_break_reclaim_v41_official_001/result_london_open_false_break_reclaim_v41.json",
        "v41_signal_action_ledger": "evidence/run_london_open_false_break_reclaim_v41_official_001/proposal_ledger_london_open_false_break_reclaim_v41.jsonl",
        "v41_official_seal": "evidence/orchestrator_state_v2/official_seal_v41.json",
        "v42_work_order": "evidence/orchestrator_state_v2/next_hypothesis_work_order_v42.json",
    }
    return {name: sha256_file(root / path) for name, path in paths.items()}


def build(root: Path) -> dict:
    policy_path = root / POLICY_PATH
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("classification") != "NON_STRATEGY_ORCHESTRATOR_EVIDENCE_POLICY" \
            or policy.get("authority") != AUTHORITY \
            or policy.get("preserves_v1_checkpoint") is not True:
        raise SignAwareEvidenceError("sign-aware policy identity or authority mismatch")
    v1.engine.frozen_v31.frozen_v30.frozen_v29.frozen_v28.runtime_v27.install_timestamp_compatibility()
    v1_payload = v1.validate(root)
    candidates = [item for item in v1_payload["candidates"] if item["minimum_gate_passed"]]
    if any(item["candidate_status"] != "RESEARCH_COMPONENT_CANDIDATE_UNADMITTED"
           or item["qualification"] != "PROVISIONAL" for item in candidates):
        raise SignAwareEvidenceError("V1 candidate admission boundary changed")
    before = binding_hashes(root)
    reviews = [pair_review(root, candidates[0], candidates[1], policy)] if len(candidates) == 2 else []
    allowed = len(candidates) >= 2 and bool(reviews) and all(item["independence_gate_passed"] for item in reviews)
    diagnostic, daily_rows = achievability_diagnostic(root, candidates, policy)
    daily_path = root / DAILY_PORTFOLIO_PATH
    v1.atomic_text(daily_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in daily_rows))
    payload = {
        "schema_version": 2,
        "registry_id": "FX_COMPONENT_WORKER_SIGN_AWARE_EVIDENCE_REGISTRY_V2",
        "classification": "NON_STRATEGY_ORCHESTRATOR_EVIDENCE",
        "policy_path": POLICY_PATH, "policy_sha256": sha256_file(policy_path),
        "v1_checkpoint_path": V1_REGISTRY_PATH,
        "v1_checkpoint_file_sha256": sha256_file(root / V1_REGISTRY_PATH),
        "v1_checkpoint_embedded_sha256": v1_payload["registry_sha256"],
        "candidate_status": "RESEARCH_COMPONENT_CANDIDATE_UNADMITTED",
        "candidate_qualification": "PROVISIONAL",
        "candidate_cycles": [item["cycle_id"] for item in candidates],
        "positive_provisional_candidate_count": len(candidates),
        "sign_aware_pair_reviews": reviews,
        "portfolio_composition_proposal_allowed": allowed,
        "portfolio_proposal_allocation_if_allowed": "FIXED_EQUAL_COMPONENT_SLEEVES",
        "achievability_diagnostic": diagnostic,
        "fixed_equal_sleeve_daily_returns_path": DAILY_PORTFOLIO_PATH,
        "fixed_equal_sleeve_daily_returns_sha256": sha256_file(daily_path),
        "strategy_adoption_authorized": False,
        "existing_profit_gate_changed": False,
        "holdout_state": "UNOPENED",
        "protected_strategy_artifact_hashes": {"before": before, "after": binding_hashes(root),
                                               "unchanged": before == binding_hashes(root)},
        "authority": AUTHORITY,
    }
    payload["registry_sha256"] = embedded_hash(payload, "registry_sha256")
    v1.atomic_text(root / REGISTRY_PATH,
                   json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return payload


def validate(root: Path) -> dict:
    path = root / REGISTRY_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("registry_sha256") != embedded_hash(payload, "registry_sha256"):
        raise SignAwareEvidenceError("V2 registry embedded hash mismatch")
    if payload.get("authority") != AUTHORITY or payload.get("strategy_adoption_authorized") is not False \
            or payload.get("existing_profit_gate_changed") is not False or payload.get("holdout_state") != "UNOPENED":
        raise SignAwareEvidenceError("V2 authority, adoption, profit gate, or holdout changed")
    policy_path = root / payload.get("policy_path", "")
    if not policy_path.is_file() or sha256_file(policy_path) != payload.get("policy_sha256"):
        raise SignAwareEvidenceError("V2 policy hash mismatch")
    if sha256_file(root / V1_REGISTRY_PATH) != payload.get("v1_checkpoint_file_sha256"):
        raise SignAwareEvidenceError("V1 component checkpoint changed")
    protected = payload.get("protected_strategy_artifact_hashes", {})
    current = binding_hashes(root)
    if protected.get("unchanged") is not True or protected.get("before") != current or protected.get("after") != current:
        raise SignAwareEvidenceError("V41 or V42 protected artifact changed")
    daily_path = root / payload.get("fixed_equal_sleeve_daily_returns_path", "")
    if not daily_path.is_file() or sha256_file(daily_path) != payload.get("fixed_equal_sleeve_daily_returns_sha256"):
        raise SignAwareEvidenceError("V2 fixed-equal-sleeve daily evidence changed")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    reviews = payload.get("sign_aware_pair_reviews", [])
    for review in reviews:
        rho = review["daily_base_return_correlation"]
        if rho < policy["correlation_routing"]["negative_complementarity_review_if_strictly_less_than"] \
                and review.get("routing") != "COMPLEMENTARITY_REVIEW":
            raise SignAwareEvidenceError("negative correlation bypassed complementarity review")
        if rho > policy["correlation_routing"]["positive_duplicate_if_strictly_greater_than"] \
                and review.get("independence_gate_passed") is not False:
            raise SignAwareEvidenceError("positive duplicate correlation passed")
        if review.get("independence_gate_passed") is not all(review["complementarity_checks"].values()) \
                and review.get("routing") != "POSITIVE_CORRELATION_DUPLICATE_REVIEW":
            raise SignAwareEvidenceError("sign-aware complementarity checks mismatch")
    allowed = len(payload.get("candidate_cycles", [])) >= 2 and bool(reviews) \
        and all(item["independence_gate_passed"] for item in reviews)
    if payload.get("portfolio_composition_proposal_allowed") is not allowed:
        raise SignAwareEvidenceError("V2 portfolio proposal gate mismatch")
    diagnostic = payload.get("achievability_diagnostic", {})
    if diagnostic.get("oracle_used") is not False \
            or diagnostic.get("post_hoc_leverage_used") is not False \
            or diagnostic.get("profit_gate_pass_inferred") is not False \
            or diagnostic.get("holdout_reproduction_inferred") is not False:
        raise SignAwareEvidenceError("achievability diagnostic overstates admission")
    return payload
