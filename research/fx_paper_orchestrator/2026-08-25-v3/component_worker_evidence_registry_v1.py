"""Build a read-only registry of unadmitted paper-research component evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

import run_asian_displacement_handoff_fade_v32 as engine


POLICY_PATH = "COMPONENT_WORKER_EVIDENCE_POLICY_V1.json"
OUTPUT_ROOT = "evidence/component_worker_registry_v1"
REGISTRY_PATH = f"{OUTPUT_ROOT}/component_worker_registry_v1.json"
INITIAL_FAILURE_PATH = "COMPONENT_WORKER_REGISTRY_INITIAL_BUILD_FAILURE.json"
INITIAL_FAILURE_SHA256 = "c7dfcd3312e87bd79c1210d7688d3fd628574542b99be548e50b9776868a485d"
SOURCE_ROOT = Path("/Users/tossaki/App/QuantRabbit/logs/replay/oanda_history/20260715T115624Z")
WALK_START = "2026-05-01"
WALK_END = "2026-07-01"
ARMS = ("RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS")
AUTHORITY = {
    "paper_only": True,
    "live_authority": False,
    "broker_account_access": False,
    "credential_access": False,
    "order_endpoint": False,
    "external_orders": 0,
    "deploy": False,
    "external_config_mutation": False,
}
CANDIDATES = (
    {
        "cycle_id": "V38",
        "family": "FX_SESSION_RANGE_NORMALIZED_MEAN_REVERSION",
        "session": "LONDON_MORNING_0800_1155_DECISION_1200_FILL",
        "horizon_seconds": 172800,
        "result": "evidence/run_london_overextension_fade_v38_official_001/result_london_overextension_fade_v38.json",
        "ledger": "evidence/run_london_overextension_fade_v38_official_001/proposal_ledger_london_overextension_fade_v38.jsonl",
        "seal": "evidence/orchestrator_state_v2/official_seal_v38.json",
    },
    {
        "cycle_id": "V40",
        "family": "FX_LONDON_FIX_MEAN_REVERSION",
        "session": "LONDON_FIX_PREWINDOW_1200_1555_DECISION_1600_FILL",
        "horizon_seconds": 14100,
        "result": "evidence/run_london_fix_overextension_fade_v40_official_001/result_london_fix_overextension_fade_v40.json",
        "ledger": "evidence/run_london_fix_overextension_fade_v40_official_001/proposal_ledger_london_fix_overextension_fade_v40.jsonl",
        "seal": "evidence/orchestrator_state_v2/official_seal_v40.json",
    },
)


class ComponentEvidenceError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedded_hash(payload: dict[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(canonical_bytes(unsigned)).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_corpus() -> dict[str, list]:
    corpus = {}
    for pair in sorted(engine.UNIVERSE):
        matches = sorted((SOURCE_ROOT / pair).glob("*_M5_BA_*.jsonl.gz"))
        if len(matches) != 1:
            raise ComponentEvidenceError(f"expected one source file for {pair}")
        corpus[pair] = engine.load_bars(matches[0])
    return corpus


def portfolio_marks_and_exposure(plans: dict[str, dict], arm: str) -> tuple[dict[str, float], list[dict]]:
    pair_marks = {}
    pair_directions = {}
    for pair, plan in sorted(plans.items()):
        pair_marks[pair], _active, pair_directions[pair], _returns = \
            engine.frozen_v31.frozen_v30.frozen_v29.frozen_v28._pair_marks(plan, arm)
    common = set.intersection(*(set(values) for values in pair_marks.values()))
    stamps = sorted(common)
    marks = {
        stamp: statistics.fmean(pair_marks[pair][stamp] for pair in sorted(engine.UNIVERSE))
        for stamp in stamps
    }
    exposure_rows = []
    for stamp in stamps:
        vector: dict[str, float] = {}
        for pair in sorted(engine.UNIVERSE):
            direction = pair_directions[pair][stamp]
            if direction == 0:
                continue
            base, quote = pair.split("_")
            signed = (1 / 7) * direction
            vector[base] = vector.get(base, 0.0) + signed
            vector[quote] = vector.get(quote, 0.0) - signed
        vector = {currency: value for currency, value in sorted(vector.items()) if value != 0.0}
        if vector:
            exposure_rows.append({"timestamp": stamp, "currency_exposure_nav": vector})
    return marks, exposure_rows


def daily_returns(arm_marks: dict[str, dict[str, float]]) -> list[dict]:
    closes: dict[str, dict[str, float]] = {}
    for arm, marks in arm_marks.items():
        for stamp, value in marks.items():
            closes.setdefault(stamp[:10], {})[arm] = value
    prior = {arm: 1.0 for arm in ARMS}
    rows = []
    for day in sorted(closes):
        if set(closes[day]) != set(ARMS):
            raise ComponentEvidenceError("daily arm calendars differ")
        arm_returns = {}
        for arm in ARMS:
            arm_returns[arm] = closes[day][arm] / prior[arm] - 1.0
            prior[arm] = closes[day][arm]
        rows.append({"utc_day": day, "returns": arm_returns})
    return rows


def hard_guard_violations(result: dict, policy: dict) -> list[str]:
    walk = result["periods"]["WALK_FORWARD"]
    violations = []
    if result.get("holdout", {}).get("state") != policy["minimum_candidate_gate"]["holdout_state"]:
        violations.append("HOLDOUT_STATE")
    if result.get("external_orders") != 0 or result.get("authority", {}).get("external_orders") != 0:
        violations.append("EXTERNAL_ORDERS")
    if result.get("live_authority") is not False or result.get("authority") != AUTHORITY:
        violations.append("AUTHORITY")
    if result.get("same_signal_stream_all_cost_arms") is not True \
            or result.get("same_execution_actions_all_cost_arms") is not True \
            or result.get("same_execution_state_transitions_all_cost_arms") is not True:
        violations.append("ARM_PARITY")
    transition_hashes = {walk[arm]["execution_state_transition_sha256"] for arm in ARMS}
    if len(transition_hashes) != 1:
        violations.append("TRANSITION_PARITY")
    for arm in ARMS:
        metrics = walk[arm]
        if metrics["terminal_open_inventory"] != 0 or metrics["terminal_inventory_mtm"] != 0.0:
            violations.append(f"TERMINAL_{arm}")
        if metrics["max_gross_exposure_nav"] > result["portfolio"]["gross_leverage_cap"] + 1e-12:
            violations.append(f"GROSS_CAP_{arm}")
        if metrics["max_currency_abs_exposure_nav"] > result["portfolio"]["currency_abs_exposure_cap"] + 1e-12:
            violations.append(f"CURRENCY_CAP_{arm}")
        if metrics["max_inventory_age_seconds"] > result["execution_rule"]["hard_max_age_seconds"]:
            violations.append(f"MAX_AGE_{arm}")
    return violations


def reconstruct_candidate(root: Path, spec: dict, policy: dict, corpus: dict[str, list]) -> dict:
    result_path = root / spec["result"]
    ledger_path = root / spec["ledger"]
    seal_path = root / spec["seal"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
    if seal.get("cycle_id") != spec["cycle_id"] or seal.get("system_acceptance", {}).get("passed") is not True:
        raise ComponentEvidenceError(f"{spec['cycle_id']} seal is not a system pass")
    if seal.get("strategy_profit_gate", {}).get("passed") is not False \
            or seal.get("strategy_profit_gate", {}).get("adoption_authorized") is not False:
        raise ComponentEvidenceError(f"{spec['cycle_id']} may not be registered as unadmitted")
    if seal.get("result_file_sha256") != sha256_file(result_path) \
            or seal.get("ledger_sha256") != sha256_file(ledger_path) \
            or result.get("result_sha256") != embedded_hash(result, "result_sha256"):
        raise ComponentEvidenceError(f"{spec['cycle_id']} sealed hashes differ")

    prior = engine.frozen_v31.TARGET_HOLD_SECONDS
    engine.frozen_v31.TARGET_HOLD_SECONDS = spec["horizon_seconds"]
    try:
        plans = engine.frozen_v31.build_period_plans(corpus, rows, WALK_START, WALK_END)
        arm_marks = {}
        exposure_rows = None
        for arm in ARMS:
            arm_marks[arm], candidate_exposure = portfolio_marks_and_exposure(plans, arm)
            if exposure_rows is None:
                exposure_rows = candidate_exposure
            elif exposure_rows != candidate_exposure:
                raise ComponentEvidenceError("cost arms changed currency-time exposure")
    finally:
        engine.frozen_v31.TARGET_HOLD_SECONDS = prior
    return_rows = daily_returns(arm_marks)
    for arm in ARMS:
        final_mark = arm_marks[arm][max(arm_marks[arm])]
        expected = result["periods"]["WALK_FORWARD"][arm]["equity_multiple"]
        if not math.isclose(final_mark, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ComponentEvidenceError(f"{spec['cycle_id']} daily reconstruction mismatch: {arm}")

    output = root / OUTPUT_ROOT
    daily_path = output / f"{spec['cycle_id'].lower()}_daily_worker_returns.jsonl"
    exposure_path = output / f"{spec['cycle_id'].lower()}_currency_time_exposure.jsonl"
    atomic_text(daily_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in return_rows))
    atomic_text(exposure_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in exposure_rows))

    walk = result["periods"]["WALK_FORWARD"]
    violations = hard_guard_violations(result, policy)
    minimum_pass = (
        not violations
        and walk["EXECUTABLE_BASE"]["equity_multiple"] > 1.0
        and walk["ADVERSE_STRESS"]["equity_multiple"] > 1.0
        and walk["RAW_SIGNAL"]["gross_edge_bps"] > 0.0
        and walk["EXECUTABLE_BASE"]["net_edge_bps"] > 0.0
        and walk["ADVERSE_STRESS"]["net_edge_bps"] > 0.0
        and walk["RAW_SIGNAL"]["direction_accuracy"] > 0.5
    )
    n_eff = walk["EXECUTABLE_BASE"]["N_eff_episodes"]
    nonprovisional = (
        minimum_pass
        and n_eff >= policy["provisional_only_gate"]["minimum_n_eff_episodes_for_nonprovisional"]
        and policy["provisional_only_gate"]["current_multiple_testing_adjustment"] != "NONE"
    )
    return {
        "cycle_id": spec["cycle_id"],
        "candidate_status": policy["candidate_status"] if minimum_pass else "NOT_A_COMPONENT_CANDIDATE",
        "qualification": "NONPROVISIONAL" if nonprovisional else "PROVISIONAL" if minimum_pass else "FAILED_MINIMUM_GATE",
        "strategy_adopted": False,
        "monthly_2x_inferred": False,
        "holdout_reproduction_inferred": False,
        "family": spec["family"],
        "session": spec["session"],
        "horizon_seconds": spec["horizon_seconds"],
        "minimum_gate_passed": minimum_pass,
        "hard_guard_violations": violations,
        "walk_forward": {
            arm: {key: walk[arm][key] for key in (
                "equity_multiple", "gross_edge_bps", "realized_cost_bps", "net_edge_bps",
                "direction_accuracy", "N_eff_days", "N_eff_episodes", "turnover_nav",
                "terminal_open_inventory", "terminal_inventory_mtm",
            )} for arm in ARMS
        },
        "evidence": {
            "official_seal_path": spec["seal"],
            "official_seal_file_sha256": sha256_file(seal_path),
            "result_path": spec["result"],
            "result_file_sha256": sha256_file(result_path),
            "signal_ledger_path": spec["ledger"],
            "signal_ledger_sha256": sha256_file(ledger_path),
            "daily_worker_returns_path": str(daily_path.relative_to(root)),
            "daily_worker_returns_sha256": sha256_file(daily_path),
            "currency_time_exposure_path": str(exposure_path.relative_to(root)),
            "currency_time_exposure_sha256": sha256_file(exposure_path),
            "n_eff_sha256": hashlib.sha256(canonical_bytes({
                "days": walk["EXECUTABLE_BASE"]["N_eff_days"], "episodes": n_eff,
            })).hexdigest(),
            "turnover_sha256": hashlib.sha256(canonical_bytes({arm: walk[arm]["turnover_nav"] for arm in ARMS})).hexdigest(),
            "cost_drag_sha256": hashlib.sha256(canonical_bytes({arm: walk[arm]["realized_cost_bps"] for arm in ARMS})).hexdigest(),
            "family_session_horizon_sha256": hashlib.sha256(canonical_bytes({
                "family": spec["family"], "session": spec["session"], "horizon_seconds": spec["horizon_seconds"],
            })).hexdigest(),
        },
    }


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ComponentEvidenceError("return correlation vectors differ")
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right))
    return numerator / denominator if denominator else 0.0


def pairwise_independence(root: Path, candidates: list[dict], policy: dict) -> list[dict]:
    result = []
    for index, left in enumerate(candidates):
        for right in candidates[index + 1:]:
            left_daily = [json.loads(line) for line in (root / left["evidence"]["daily_worker_returns_path"]).read_text().splitlines()]
            right_daily = [json.loads(line) for line in (root / right["evidence"]["daily_worker_returns_path"]).read_text().splitlines()]
            days = sorted({row["utc_day"] for row in left_daily} | {row["utc_day"] for row in right_daily})
            left_by_day = {row["utc_day"]: row["returns"]["EXECUTABLE_BASE"] for row in left_daily}
            right_by_day = {row["utc_day"]: row["returns"]["EXECUTABLE_BASE"] for row in right_daily}
            correlation = pearson([left_by_day.get(day, 0.0) for day in days],
                                  [right_by_day.get(day, 0.0) for day in days])
            def clusters(candidate: dict) -> set[tuple[str, str, int]]:
                rows = [json.loads(line) for line in (root / candidate["evidence"]["currency_time_exposure_path"]).read_text().splitlines()]
                return {(row["timestamp"], currency, 1 if value > 0 else -1)
                        for row in rows for currency, value in row["currency_exposure_nav"].items()}
            left_clusters, right_clusters = clusters(left), clusters(right)
            overlap = len(left_clusters & right_clusters) / len(left_clusters | right_clusters) \
                if left_clusters | right_clusters else 0.0
            same_session = left["session"] == right["session"]
            passed = (
                abs(correlation) <= policy["independence_gate"]["maximum_absolute_daily_base_return_correlation"]
                and overlap <= policy["independence_gate"]["maximum_signed_currency_time_jaccard_overlap"]
                and not same_session
            )
            result.append({
                "left_cycle": left["cycle_id"], "right_cycle": right["cycle_id"],
                "calendar_days": len(days), "daily_base_return_correlation": correlation,
                "signed_currency_time_cluster_overlap": overlap,
                "left_currency_time_clusters": len(left_clusters),
                "right_currency_time_clusters": len(right_clusters),
                "shared_currency_time_clusters": len(left_clusters & right_clusters),
                "same_session": same_session, "independence_gate_passed": passed,
            })
    return result


def build(root: Path) -> dict:
    policy_path = root / POLICY_PATH
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("classification") != "NON_STRATEGY_ORCHESTRATOR_EVIDENCE_POLICY" \
            or policy.get("authority") != AUTHORITY:
        raise ComponentEvidenceError("component policy authority or classification mismatch")
    failure_path = root / INITIAL_FAILURE_PATH
    if not failure_path.is_file() or sha256_file(failure_path) != INITIAL_FAILURE_SHA256:
        raise ComponentEvidenceError("initial component build failure evidence changed")
    v41_result = root / "evidence/run_london_open_false_break_reclaim_v41_official_001/result_london_open_false_break_reclaim_v41.json"
    v41_ledger = root / "evidence/run_london_open_false_break_reclaim_v41_official_001/proposal_ledger_london_open_false_break_reclaim_v41.jsonl"
    v41_seal = root / "evidence/orchestrator_state_v2/official_seal_v41.json"
    v41_before = {
        "result": sha256_file(v41_result), "ledger": sha256_file(v41_ledger),
        "official_seal": sha256_file(v41_seal),
    }
    engine.frozen_v31.frozen_v30.frozen_v29.frozen_v28.runtime_v27.install_timestamp_compatibility()
    corpus = load_corpus()
    candidates = [reconstruct_candidate(root, spec, policy, corpus) for spec in CANDIDATES]
    pairs = pairwise_independence(root, candidates, policy)
    positive = [candidate for candidate in candidates if candidate["minimum_gate_passed"]]
    independence_passed = (
        len(positive) >= policy["independence_gate"]["minimum_positive_provisional_candidates"]
        and bool(pairs) and all(pair["independence_gate_passed"] for pair in pairs)
    )
    payload = {
        "schema_version": 1,
        "registry_id": "FX_COMPONENT_WORKER_EVIDENCE_REGISTRY_V1",
        "classification": "NON_STRATEGY_ORCHESTRATOR_EVIDENCE",
        "generated_after_cycle": "V41",
        "policy_path": POLICY_PATH,
        "policy_sha256": sha256_file(policy_path),
        "initial_build_failure_evidence": {
            "path": INITIAL_FAILURE_PATH, "sha256": INITIAL_FAILURE_SHA256,
            "status": "RECOVERED_RESTART_SAFE_WITHOUT_STRATEGY_RERUN",
        },
        "source_mode": "DIRECT_READBACK_FROM_IMMUTABLE_SEALED_RESULTS_AND_LEDGERS",
        "evaluation_calendar": {"start": WALK_START, "end_exclusive": WALK_END},
        "cost_assumptions": "FROZEN_V7_RAW_BASE_ADVERSE_ARMS",
        "candidate_count": len(candidates),
        "positive_provisional_candidate_count": len(positive),
        "candidates": candidates,
        "deduplicated_variants": [{
            "cycle_id": "V39", "canonical_component_cycle": "V38",
            "reason": "SAME_SIGNAL_FAMILY_WEAKER_CARRY_VARIANT_NOT_AN_INDEPENDENT_WORKER",
            "counted_as_independent_worker": False,
        }],
        "pairwise_currency_time_independence": pairs,
        "portfolio_composition_proposal_allowed": independence_passed,
        "portfolio_proposal_rule_if_allowed": "FIXED_EQUAL_COMPONENT_SLEEVES",
        "existing_profit_gate_changed": False,
        "strategy_adoption_authorized": False,
        "v41_artifact_hashes_before_and_after": {
            "before": v41_before,
            "after": {
                "result": sha256_file(v41_result), "ledger": sha256_file(v41_ledger),
                "official_seal": sha256_file(v41_seal),
            },
            "unchanged": v41_before == {
                "result": sha256_file(v41_result), "ledger": sha256_file(v41_ledger),
                "official_seal": sha256_file(v41_seal),
            },
        },
        "authority": AUTHORITY,
    }
    payload["registry_sha256"] = embedded_hash(payload, "registry_sha256")
    atomic_text(root / REGISTRY_PATH, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return payload


def validate(root: Path) -> dict:
    path = root / REGISTRY_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("registry_sha256") != embedded_hash(payload, "registry_sha256"):
        raise ComponentEvidenceError("component registry embedded hash mismatch")
    if payload.get("authority") != AUTHORITY or payload.get("existing_profit_gate_changed") is not False:
        raise ComponentEvidenceError("component registry changed authority or profit gate")
    if payload.get("strategy_adoption_authorized") is not False \
            or payload.get("v41_artifact_hashes_before_and_after", {}).get("unchanged") is not True:
        raise ComponentEvidenceError("component registry implies adoption or changed V41")
    policy_path = root / payload.get("policy_path", "")
    if not policy_path.is_file() or sha256_file(policy_path) != payload.get("policy_sha256"):
        raise ComponentEvidenceError("component policy hash mismatch")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    failure = payload.get("initial_build_failure_evidence", {})
    failure_path = root / failure.get("path", "")
    if failure.get("sha256") != INITIAL_FAILURE_SHA256 \
            or not failure_path.is_file() or sha256_file(failure_path) != INITIAL_FAILURE_SHA256 \
            or failure.get("status") != "RECOVERED_RESTART_SAFE_WITHOUT_STRATEGY_RERUN":
        raise ComponentEvidenceError("initial component build failure is hidden or changed")
    v41_expected = {
        "result": sha256_file(root / "evidence/run_london_open_false_break_reclaim_v41_official_001/result_london_open_false_break_reclaim_v41.json"),
        "ledger": sha256_file(root / "evidence/run_london_open_false_break_reclaim_v41_official_001/proposal_ledger_london_open_false_break_reclaim_v41.jsonl"),
        "official_seal": sha256_file(root / "evidence/orchestrator_state_v2/official_seal_v41.json"),
    }
    v41_binding = payload["v41_artifact_hashes_before_and_after"]
    if v41_binding.get("before") != v41_expected or v41_binding.get("after") != v41_expected:
        raise ComponentEvidenceError("V41 result, ledger, or seal binding changed")
    positive = 0
    for candidate in payload.get("candidates", []):
        for path_key, hash_key in (
            ("signal_ledger_path", "signal_ledger_sha256"),
            ("daily_worker_returns_path", "daily_worker_returns_sha256"),
            ("currency_time_exposure_path", "currency_time_exposure_sha256"),
        ):
            artifact = root / candidate["evidence"][path_key]
            if not artifact.is_file() or sha256_file(artifact) != candidate["evidence"][hash_key]:
                raise ComponentEvidenceError(f"component evidence changed: {candidate['cycle_id']} {path_key}")
        walk = candidate.get("walk_forward", {})
        minimum_pass = (
            candidate.get("hard_guard_violations") == []
            and walk.get("EXECUTABLE_BASE", {}).get("equity_multiple", 0) > 1.0
            and walk.get("ADVERSE_STRESS", {}).get("equity_multiple", 0) > 1.0
            and walk.get("RAW_SIGNAL", {}).get("gross_edge_bps", 0) > 0.0
            and walk.get("EXECUTABLE_BASE", {}).get("net_edge_bps", 0) > 0.0
            and walk.get("ADVERSE_STRESS", {}).get("net_edge_bps", 0) > 0.0
            and walk.get("RAW_SIGNAL", {}).get("direction_accuracy", 0) > 0.5
        )
        if candidate.get("minimum_gate_passed") is not minimum_pass:
            raise ComponentEvidenceError(f"component minimum gate mismatch: {candidate.get('cycle_id')}")
        if minimum_pass:
            positive += 1
            if candidate.get("candidate_status") != policy["candidate_status"] \
                    or candidate.get("qualification") != "PROVISIONAL" \
                    or candidate.get("strategy_adopted") is not False \
                    or candidate.get("monthly_2x_inferred") is not False \
                    or candidate.get("holdout_reproduction_inferred") is not False:
                raise ComponentEvidenceError(f"component admission boundary changed: {candidate.get('cycle_id')}")
    if positive != payload.get("positive_provisional_candidate_count"):
        raise ComponentEvidenceError("positive provisional candidate count mismatch")
    dedup = payload.get("deduplicated_variants", [])
    if not any(item.get("cycle_id") == "V39" and item.get("canonical_component_cycle") == "V38"
               and item.get("counted_as_independent_worker") is False for item in dedup):
        raise ComponentEvidenceError("V39 duplicate-family exclusion is missing")
    pairs = payload.get("pairwise_currency_time_independence", [])
    allowed = (
        positive >= policy["independence_gate"]["minimum_positive_provisional_candidates"]
        and bool(pairs) and all(pair.get("independence_gate_passed") is True for pair in pairs)
    )
    if payload.get("portfolio_composition_proposal_allowed") is not allowed:
        raise ComponentEvidenceError("portfolio proposal gate mismatch")
    return payload
