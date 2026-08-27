"""Build a sealed-evidence cause and feasibility audit for the monthly 2x gate.

This module is read-only with respect to every historical strategy artifact. It
uses the separately hash-fixed 13-seal derived pair audit as primary pair
evidence and the frozen V38/V40 component registry for exposure decomposition.
No diagnostic in this file can admit a strategy or authorize an official run.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
COMPAT_DIR = MODULE_DIR / "paper_replay_compat"
for import_root in (MODULE_DIR, COMPAT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import component_worker_evidence_registry_v1 as component_v1
import component_worker_evidence_registry_v2 as component_v2
import derived_pair_audit_runner_v1 as derived_pair


POLICY_PATH = "PROFIT_GATE_CAUSE_FEASIBILITY_POLICY_V1.json"
OUTPUT_ROOT = "evidence/profit_gate_cause_feasibility_v1"
AUDIT_PATH = f"{OUTPUT_ROOT}/profit_gate_cause_feasibility_audit_v1.json"
PAIR_READBACK_PATH = derived_pair.METRICS_PATH
AUD_CROSS_INVENTORY_PATH = f"{OUTPUT_ROOT}/aud_cross_local_corpus_inventory.jsonl"
SOURCE_ROOT = component_v1.SOURCE_ROOT
ARMS = component_v1.ARMS
AUTHORITY = component_v1.AUTHORITY
TARGET_LOG_GROWTH = math.log(2.0)
PAIR_SLEEVE = 1.0 / 7.0
M5_SECONDS = 300
SELECTED_PAIR_ARMS = ("USD_JPY", "EUR_USD", "AUD_USD")
LEGACY_LLM_V15_ROOT = Path(
    "/Users/tossaki/.codex/worktrees/1e7a/QuantRabbit/"
    "research/llm_paper_experiment/2026-08-21-v15"
)
LEGACY_SOURCE_REPO = Path("/Users/tossaki/.codex/worktrees/1e7a/QuantRabbit")
LEGACY_LLM_V18_ROOT = (
    LEGACY_SOURCE_REPO / "research/llm_paper_experiment/2026-08-21-v18"
)
LEGACY_LLM_V253_ROOT = (
    LEGACY_SOURCE_REPO / "research/llm_paper_experiment/2026-08-24-v253"
)
LEGACY_LLM_V252_ROOT = (
    LEGACY_SOURCE_REPO / "research/llm_paper_experiment/2026-08-24-v252"
)
LEGACY_LLM_V250_ROOT = (
    LEGACY_SOURCE_REPO / "research/llm_paper_experiment/2026-08-24-v250"
)
AI_TEST_BOT_ROOT = Path("/Users/tossaki/App/QuantRabbit")
BROKER_COHORT_REPORT = (
    LEGACY_SOURCE_REPO / "research/historical_learning_price_action_admission/report_v1.json"
)
PERIODS = {
    "TUNING_2026_03_11_TO_2026_05_01": ("2026-03-11", "2026-05-01"),
    "MONTH_2026_05": ("2026-05-01", "2026-06-01"),
    "MONTH_2026_06": ("2026-06-01", "2026-07-01"),
    "WALK_FORWARD": ("2026-05-01", "2026-07-01"),
}
MONTHS = ("MONTH_2026_05", "MONTH_2026_06")
CANDIDATES = {item["cycle_id"]: item for item in component_v1.CANDIDATES}
PROTECTED_PATHS = {
    "v38_result": CANDIDATES["V38"]["result"],
    "v38_signal_ledger": CANDIDATES["V38"]["ledger"],
    "v38_official_seal": CANDIDATES["V38"]["seal"],
    "v40_result": CANDIDATES["V40"]["result"],
    "v40_signal_ledger": CANDIDATES["V40"]["ledger"],
    "v40_official_seal": CANDIDATES["V40"]["seal"],
    "v41_result": (
        "evidence/run_london_open_false_break_reclaim_v41_official_001/"
        "result_london_open_false_break_reclaim_v41.json"
    ),
    "v41_signal_ledger": (
        "evidence/run_london_open_false_break_reclaim_v41_official_001/"
        "proposal_ledger_london_open_false_break_reclaim_v41.jsonl"
    ),
    "v41_official_seal": "evidence/orchestrator_state_v2/official_seal_v41.json",
    "v42_work_order": "evidence/orchestrator_state_v2/next_hypothesis_work_order_v42.json",
    "component_registry_v2": component_v2.REGISTRY_PATH,
    "derived_pair_audit": derived_pair.AUDIT_PATH,
    "derived_pair_metrics": derived_pair.METRICS_PATH,
    "derived_daily_range_inputs": derived_pair.DAILY_RANGE_PATH,
    "v13_result": (
        "evidence/run_actual_llm_stale_unwind_v13_001/"
        "result_actual_llm_stale_unwind_v13.json"
    ),
    "v13_ledger": (
        "evidence/run_actual_llm_stale_unwind_v13_001/"
        "proposal_ledger_actual_llm_stale_unwind_v13.jsonl"
    ),
    "v13_decision": "ACTUAL_LLM_INVENTORY_DECISION_V13.json",
    "v13_preregistration": "ACTUAL_LLM_STALE_UNWIND_PREREGISTRATION_V13.json",
}


class CauseFeasibilityError(RuntimeError):
    """A fail-closed audit contract violation."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    return component_v1.sha256_file(path)


def embedded_hash(payload: dict[str, Any], field: str) -> str:
    return component_v1.embedded_hash(payload, field)


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def calendar_days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def protected_hashes(root: Path) -> dict[str, str]:
    return {name: sha256_file(root / relative) for name, relative in PROTECTED_PATHS.items()}


def verify_sealed_cycle(root: Path, spec: dict[str, Any]) -> tuple[dict, list[dict]]:
    result_path = root / spec["result"]
    ledger_path = root / spec["ledger"]
    seal_path = root / spec["seal"]
    result, seal = load_json(result_path), load_json(seal_path)
    rows = load_jsonl(ledger_path)
    if result.get("cycle_id") != spec["cycle_id"] or seal.get("cycle_id") != spec["cycle_id"]:
        raise CauseFeasibilityError(f"sealed identity mismatch: {spec['cycle_id']}")
    if result.get("result_sha256") != embedded_hash(result, "result_sha256"):
        raise CauseFeasibilityError(f"embedded result hash mismatch: {spec['cycle_id']}")
    if seal.get("result_file_sha256") != sha256_file(result_path) \
            or seal.get("ledger_sha256") != sha256_file(ledger_path):
        raise CauseFeasibilityError(f"official seal readback mismatch: {spec['cycle_id']}")
    if result.get("holdout", {}).get("state") != "UNOPENED" \
            or result.get("authority") != AUTHORITY or result.get("external_orders") != 0:
        raise CauseFeasibilityError(f"holdout or authority boundary changed: {spec['cycle_id']}")
    if seal.get("strategy_profit_gate", {}).get("passed") is not False \
            or seal.get("strategy_profit_gate", {}).get("adoption_authorized") is not False:
        raise CauseFeasibilityError(f"historical strategy unexpectedly admitted: {spec['cycle_id']}")
    for arm in ARMS:
        walk = result["periods"]["WALK_FORWARD"][arm]
        if walk["terminal_open_inventory"] != 0 or walk["terminal_inventory_mtm"] != 0.0:
            raise CauseFeasibilityError(f"terminal inventory is nonzero: {spec['cycle_id']} {arm}")
    return result, rows


def verify_source_files(results: dict[str, dict]) -> list[dict[str, Any]]:
    expected_by_cycle = {
        cycle: {item["pair"]: item for item in result["source_audit"]}
        for cycle, result in results.items()
    }
    if expected_by_cycle["V38"] != expected_by_cycle["V40"]:
        raise CauseFeasibilityError("V38/V40 source audits differ")
    records = []
    for pair, expected in sorted(expected_by_cycle["V38"].items()):
        matches = sorted((SOURCE_ROOT / pair).glob("*_M5_BA_*.jsonl.gz"))
        if len(matches) != 1:
            raise CauseFeasibilityError(f"expected one completed BID/ASK source for {pair}")
        actual_hash = sha256_file(matches[0])
        if actual_hash != expected["source_sha256"]:
            raise CauseFeasibilityError(f"source hash changed for {pair}")
        records.append({
            "pair": pair,
            "path": str(matches[0]),
            "source_sha256": actual_hash,
            "sealed_bar_count": expected["bars"],
            "direct_hash_readback_passed": True,
        })
    return records


def max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1.0)
    return drawdown


def pair_metrics(plan: dict[str, Any], spec: dict[str, Any], period: str,
                 start: str, end: str) -> dict[str, Any]:
    mark_function = (
        component_v1.engine.frozen_v31.frozen_v30.frozen_v29.frozen_v28._pair_marks
    )
    marks_by_arm: dict[str, dict[str, float]] = {}
    returns_by_arm: dict[str, list[float]] = {}
    for arm in ARMS:
        marks_by_arm[arm], _active, _directions, returns_by_arm[arm] = mark_function(plan, arm)
    episode_count = len(plan["episodes"])
    if any(len(returns_by_arm[arm]) != episode_count for arm in ARMS):
        raise CauseFeasibilityError("pair arm episode counts differ")
    signal_days = len({row["utc_day"] for row in plan["signals"]})
    days = calendar_days(start, end)
    arm_payload = {}
    raw_returns = returns_by_arm["RAW_SIGNAL"]
    for arm in ARMS:
        returns = returns_by_arm[arm]
        gross_bps = statistics.fmean(raw_returns) * 10000.0 if raw_returns else None
        net_bps = statistics.fmean(returns) * 10000.0 if returns else None
        cost_bps = statistics.fmean(gross - net for gross, net in zip(raw_returns, returns)) * 10000.0 \
            if returns else None
        expected_move_ratio = cost_bps / gross_bps if gross_bps is not None and gross_bps > 0 else None
        values = [marks_by_arm[arm][stamp] for stamp in sorted(
            marks_by_arm[arm], key=component_v1.engine.frozen_v31.ns
        )]
        multiple = values[-1]
        arm_payload[arm] = {
            "pair_standalone_equity_multiple": multiple,
            "fixed_one_seventh_capital_contribution_multiple": 1.0 + (multiple - 1.0) * PAIR_SLEEVE,
            "log_growth": math.log(multiple),
            "gross_edge_bps": gross_bps,
            "realized_cost_bps": cost_bps,
            "net_edge_bps": net_bps,
            "cost_to_expected_move_ratio": expected_move_ratio,
            "expected_move_ratio_status": "DEFINED_POSITIVE_GROSS_EDGE" if expected_move_ratio is not None
            else "UNDEFINED_NONPOSITIVE_OR_NO_GROSS_EDGE",
            "max_drawdown": max_drawdown(values),
            "terminal_inventory_mtm": 0.0,
            "terminal_open_inventory": 0,
        }
    actions = Counter(event["action"] for event in plan["signal_events"])
    return {
        "cycle_id": spec["cycle_id"],
        "period": period,
        "start": start,
        "end_exclusive": end,
        "calendar_days": days,
        "pair": plan["pair"],
        "family": spec["family"],
        "session": spec["session"],
        "source_timeframe": "M5_COMPLETED_BID_ASK",
        "official_or_diagnostic": "OFFICIAL_SEALED_PERIOD_READBACK" if period != "TUNING_2026_03_11_TO_2026_05_01"
        else "TRAINING_ONLY_READ_ONLY_DIAGNOSTIC",
        "source_signals": len(plan["signals"]),
        "N_eff_signal_days": signal_days,
        "N_eff_episodes": episode_count,
        "signals_per_calendar_day": len(plan["signals"]) / days,
        "episodes_per_calendar_day": episode_count / days,
        "episodes_per_N_eff_signal_day": episode_count / signal_days if signal_days else 0.0,
        "turnover_nav_at_fixed_one_seventh": 2.0 * episode_count * PAIR_SLEEVE,
        "max_inventory_age_seconds": max(
            [0.0] + [episode["inventory_age_seconds"] for episode in plan["episodes"]]
        ),
        "action_counts": dict(sorted(actions.items())),
        "tested_status": "TESTED_WITH_REALIZED_EPISODES" if episode_count else "NO_REALIZED_EPISODE",
        "arms": arm_payload,
    }


def reconstruct_pair_readback(root: Path, results: dict[str, dict],
                              ledgers: dict[str, list[dict]]) -> tuple[list[dict], dict]:
    component_v1.engine.frozen_v31.frozen_v30.frozen_v29.frozen_v28.runtime_v27 \
        .install_timestamp_compatibility()
    corpus = component_v1.load_corpus()
    rows: list[dict[str, Any]] = []
    official_reconciliation = {}
    for cycle, spec in sorted(CANDIDATES.items()):
        prior = component_v1.engine.frozen_v31.TARGET_HOLD_SECONDS
        component_v1.engine.frozen_v31.TARGET_HOLD_SECONDS = spec["horizon_seconds"]
        try:
            for period, (start, end) in PERIODS.items():
                plans = component_v1.engine.frozen_v31.build_period_plans(
                    corpus, ledgers[cycle], start, end
                )
                period_rows = [pair_metrics(plan, spec, period, start, end)
                               for _pair, plan in sorted(plans.items())]
                rows.extend(period_rows)
                if period in results[cycle]["periods"]:
                    reconciliation = {}
                    for arm in ARMS:
                        reconstructed = statistics.fmean(
                            item["arms"][arm]["pair_standalone_equity_multiple"]
                            for item in period_rows
                        )
                        sealed = results[cycle]["periods"][period][arm]["equity_multiple"]
                        if not math.isclose(reconstructed, sealed, rel_tol=0.0, abs_tol=1e-12):
                            raise CauseFeasibilityError(
                                f"pair reconstruction differs from {cycle} {period} {arm}"
                            )
                        reconciliation[arm] = {
                            "reconstructed": reconstructed,
                            "sealed": sealed,
                            "absolute_difference": abs(reconstructed - sealed),
                        }
                    official_reconciliation[f"{cycle}:{period}"] = reconciliation
        finally:
            component_v1.engine.frozen_v31.TARGET_HOLD_SECONDS = prior
    return rows, official_reconciliation


def aud_cross_inventory() -> list[dict[str, Any]]:
    rows = []
    for directory in sorted(path for path in SOURCE_ROOT.iterdir() if path.is_dir()):
        pair = directory.name
        if "AUD" not in pair.split("_") or pair == "AUD_USD" or pair in component_v1.engine.UNIVERSE:
            continue
        matches = sorted(directory.glob("*_M5_BA_*.jsonl.gz"))
        if len(matches) != 1:
            raise CauseFeasibilityError(f"ambiguous AUD-cross source: {pair}")
        count = completed = 0
        first_time = last_time = None
        granularity = price = None
        with gzip.open(matches[0], "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                count += 1
                completed += item.get("complete") is True
                first_time = first_time or item.get("time")
                last_time = item.get("time")
                granularity, price = item.get("granularity"), item.get("price")
        rows.append({
            "pair": pair,
            "path": str(matches[0]),
            "compressed_file_sha256": sha256_file(matches[0]),
            "rows": count,
            "completed_rows": completed,
            "all_rows_completed": completed == count,
            "first_time": first_time,
            "last_time": last_time,
            "granularity": granularity,
            "price_component": price,
            "classification": "LOCAL_COMPLETED_BID_ASK_PRESENT_UNTESTED_SEPARATE_FAMILY",
        })
    return rows


def filter_exposure(exposures: dict[str, dict[str, float]], start: str,
                    end: str) -> dict[str, dict[str, float]]:
    return {stamp: vector for stamp, vector in exposures.items() if start <= stamp[:10] < end}


def gross_pair_exposure(vector: dict[str, float]) -> float:
    # In the frozen seven-pair USD-star universe each non-USD node maps to one
    # unique pair sleeve, so leaf absolute exposure is the non-netted pair gross.
    return sum(abs(value) for currency, value in vector.items() if currency != "USD")


def exposure_metrics(exposures: dict[str, dict[str, float]], start: str, end: str) -> dict:
    subset = filter_exposure(exposures, start, end)
    clusters = component_v2.currency_time_clusters(subset)
    gross = [gross_pair_exposure(vector) for vector in subset.values()]
    return {
        "currency_time_cluster_N_eff": len(clusters),
        "cluster_starts": [item["start"] for item in clusters],
        "completed_m5_active_rows": len(subset),
        "completed_m5_active_exposure_seconds": len(subset) * M5_SECONDS,
        "completed_m5_active_exposure_hours": len(subset) * M5_SECONDS / 3600.0,
        "completed_m5_nav_weighted_gross_exposure_seconds": sum(gross) * M5_SECONDS,
        "completed_m5_nav_weighted_gross_exposure_hours": sum(gross) * M5_SECONDS / 3600.0,
        "calendar_active_fraction_completed_bar_measure": (
            len(subset) * M5_SECONDS / (calendar_days(start, end) * 86400.0)
        ),
        "max_reconstructed_gross_exposure_nav": max([0.0] + gross),
        "max_currency_abs_exposure_nav": max(
            [0.0] + [abs(value) for vector in subset.values() for value in vector.values()]
        ),
        "weekend_wall_clock_time_included": False,
        "weekend_and_wall_clock_dwell_reported_separately_by_max_inventory_age": True,
    }


def combination_exposure_metrics(exposure_maps: dict[str, dict[str, dict[str, float]]],
                                 start: str, end: str) -> dict:
    filtered = {cycle: filter_exposure(values, start, end)
                for cycle, values in exposure_maps.items()}
    stamps = sorted(set().union(*(set(values) for values in filtered.values())),
                    key=component_v1.engine.frozen_v31.ns)
    max_gross = max_currency = 0.0
    weighted_seconds = 0.0
    for stamp in stamps:
        combined: dict[str, float] = {}
        gross = 0.0
        for exposures in filtered.values():
            vector = exposures.get(stamp, {})
            gross += 0.5 * gross_pair_exposure(vector)
            for currency, value in vector.items():
                combined[currency] = combined.get(currency, 0.0) + 0.5 * value
        max_gross = max(max_gross, gross)
        max_currency = max([max_currency] + [abs(value) for value in combined.values()])
        weighted_seconds += gross * M5_SECONDS
    individual_clusters = {
        cycle: component_v2.currency_time_clusters(exposures)
        for cycle, exposures in filtered.items()
    }
    combined_keys = {
        (item["start"], json.dumps(item["currency_exposure_nav"], sort_keys=True))
        for clusters in individual_clusters.values() for item in clusters
    }
    individual_sum = sum(len(clusters) for clusters in individual_clusters.values())
    return {
        "currency_time_cluster_N_eff": len(combined_keys),
        "individual_currency_time_clusters": {
            cycle: len(clusters) for cycle, clusters in individual_clusters.items()
        },
        "sum_individual_currency_time_clusters": individual_sum,
        "dependency_overlap_factor": len(combined_keys) / individual_sum if individual_sum else 0.0,
        "completed_m5_active_rows": len(stamps),
        "completed_m5_active_exposure_seconds": len(stamps) * M5_SECONDS,
        "completed_m5_active_exposure_hours": len(stamps) * M5_SECONDS / 3600.0,
        "completed_m5_nav_weighted_gross_exposure_seconds": weighted_seconds,
        "completed_m5_nav_weighted_gross_exposure_hours": weighted_seconds / 3600.0,
        "calendar_active_fraction_completed_bar_measure": (
            len(stamps) * M5_SECONDS / (calendar_days(start, end) * 86400.0)
        ),
        "max_reconstructed_gross_exposure_nav": max_gross,
        "max_currency_abs_exposure_nav": max_currency,
        "weekend_wall_clock_time_included": False,
        "weekend_and_wall_clock_dwell_reported_separately_by_max_inventory_age": True,
    }


def compound(values: list[float]) -> float:
    result = 1.0
    for value in values:
        result *= 1.0 + value
    return result


def daily_arm_metrics(rows: list[dict], arm: str) -> dict:
    returns = [row["returns"][arm] for row in rows]
    wealth = 1.0
    path = [wealth]
    for value in returns:
        wealth *= 1.0 + value
        path.append(wealth)
    return {"equity_multiple": wealth, "log_growth": math.log(wealth),
            "daily_mark_max_drawdown": max_drawdown(path)}


def decomposition(workers: list[dict], combined_clusters: int,
                  individual_cluster_sum: int, actual_multiples: dict[str, float]) -> dict:
    payload = {}
    weighted_episodes = sum(
        item["allocation"] * item["metrics"]["RAW_SIGNAL"]["N_eff_episodes"]
        for item in workers
    )
    if combined_clusters <= 0 or weighted_episodes <= 0:
        raise CauseFeasibilityError("decomposition lacks independent clusters or episodes")
    effective_risk = weighted_episodes * PAIR_SLEEVE / combined_clusters
    for arm in ARMS:
        gross_numerator = sum(
            item["allocation"] * item["metrics"][arm]["N_eff_episodes"]
            * item["metrics"][arm]["gross_edge_bps"] for item in workers
        )
        cost_numerator = sum(
            item["allocation"] * item["metrics"][arm]["N_eff_episodes"]
            * item["metrics"][arm]["realized_cost_bps"] for item in workers
        )
        edge_bps = gross_numerator / weighted_episodes
        cost_bps = cost_numerator / weighted_episodes
        gross_proxy = combined_clusters * (edge_bps / 10000.0) * effective_risk
        cost_proxy = combined_clusters * (cost_bps / 10000.0) * effective_risk
        net_proxy = gross_proxy - cost_proxy
        actual_log = math.log(actual_multiples[arm])
        payload[arm] = {
            "N_eff_currency_time_clusters": combined_clusters,
            "edge_per_episode_bps": edge_bps,
            "cost_per_episode_bps": cost_bps,
            "effective_risk_nav_per_independent_cluster": effective_risk,
            "dependency_overlap_factor": (
                combined_clusters / individual_cluster_sum if individual_cluster_sum else 0.0
            ),
            "gross_proxy_log_growth": gross_proxy,
            "cost_proxy_log_drag": cost_proxy,
            "net_proxy_log_growth": net_proxy,
            "actual_log_growth": actual_log,
            "actual_cost_log_drag": math.log(actual_multiples["RAW_SIGNAL"]) - actual_log,
            "dependency_tail_path_residual": actual_log - net_proxy,
            "reconciliation_identity": (
                net_proxy + (actual_log - net_proxy)
            ),
        }
    return payload


def component_month_payload(cycle: str, result: dict, month: str, exposure: dict) -> dict:
    start, end = PERIODS[month]
    period = result["periods"][month]
    multiples = {arm: period[arm]["equity_multiple"] for arm in ARMS}
    workers = [{"allocation": 1.0, "metrics": period}]
    clusters = exposure["currency_time_cluster_N_eff"]
    return {
        "calendar": {"start": start, "end_exclusive": end,
                     "calendar_days": calendar_days(start, end)},
        "N_eff": {
            "currency_time_clusters": clusters,
            "sealed_signal_days": period["RAW_SIGNAL"]["N_eff_days"],
            "sealed_realized_episodes": period["RAW_SIGNAL"]["N_eff_episodes"],
            "ticket_count_not_used_as_independent_bets": True,
        },
        "trade_density": {
            "clusters_per_calendar_day": clusters / calendar_days(start, end),
            "episodes_per_calendar_day": (
                period["RAW_SIGNAL"]["N_eff_episodes"] / calendar_days(start, end)
            ),
            "turnover_nav": period["RAW_SIGNAL"]["turnover_nav"],
        },
        "active_exposure": exposure,
        "risk": {
            "gross_cap": result["portfolio"]["gross_leverage_cap"],
            "currency_cap": result["portfolio"]["currency_abs_exposure_cap"],
            "max_gross_exposure_nav": period["EXECUTABLE_BASE"]["max_gross_exposure_nav"],
            "max_currency_abs_exposure_nav": period["EXECUTABLE_BASE"]["max_currency_abs_exposure_nav"],
            "max_margin_requirement_jpy_at_1x": period["EXECUTABLE_BASE"]["max_margin_requirement_jpy_at_1x"],
            "max_inventory_age_seconds": period["EXECUTABLE_BASE"]["max_inventory_age_seconds"],
        },
        "arms": {
            arm: {
                "equity_multiple": multiples[arm],
                "log_growth": math.log(multiples[arm]),
                "gross_edge_bps": period[arm]["gross_edge_bps"],
                "realized_cost_bps": period[arm]["realized_cost_bps"],
                "net_edge_bps": period[arm]["net_edge_bps"],
                "cost_to_expected_move_ratio": (
                    period[arm]["realized_cost_bps"] / period[arm]["gross_edge_bps"]
                    if period[arm]["gross_edge_bps"] > 0 else None
                ),
                "direction_accuracy": period[arm]["direction_accuracy"],
                "max_drawdown": period[arm]["max_drawdown"],
                "terminal_inventory_mtm": period[arm]["terminal_inventory_mtm"],
                "terminal_open_inventory": period[arm]["terminal_open_inventory"],
            } for arm in ARMS
        },
        "decomposition": decomposition(workers, clusters, clusters, multiples),
        "cycle_id": cycle,
    }


def combination_month_payload(results: dict[str, dict], month: str, exposure: dict,
                              daily_rows: list[dict]) -> dict:
    start, end = PERIODS[month]
    rows = [row for row in daily_rows if start <= row["utc_day"] < end]
    daily_metrics = {arm: daily_arm_metrics(rows, arm) for arm in ARMS}
    multiples = {arm: daily_metrics[arm]["equity_multiple"] for arm in ARMS}
    workers = [{
        "allocation": 0.5,
        "metrics": results[cycle]["periods"][month],
    } for cycle in ("V38", "V40")]
    clusters = exposure["currency_time_cluster_N_eff"]
    decomposition_payload = decomposition(
        workers, clusters, exposure["sum_individual_currency_time_clusters"], multiples
    )
    max_age = max(
        results[cycle]["periods"][month]["EXECUTABLE_BASE"]["max_inventory_age_seconds"]
        for cycle in ("V38", "V40")
    )
    return {
        "calendar": {"start": start, "end_exclusive": end,
                     "calendar_days": calendar_days(start, end)},
        "N_eff": {
            "currency_time_clusters": clusters,
            "individual_clusters": exposure["individual_currency_time_clusters"],
            "dependency_overlap_factor": exposure["dependency_overlap_factor"],
            "ticket_count_not_used_as_independent_bets": True,
        },
        "trade_density": {
            "clusters_per_calendar_day": clusters / calendar_days(start, end),
            "turnover_nav": statistics.fmean(
                results[cycle]["periods"][month]["RAW_SIGNAL"]["turnover_nav"]
                for cycle in ("V38", "V40")
            ),
        },
        "active_exposure": exposure,
        "risk": {
            "gross_cap": 1.0,
            "currency_cap": 1.0,
            "max_gross_exposure_nav": exposure["max_reconstructed_gross_exposure_nav"],
            "max_currency_abs_exposure_nav": exposure["max_currency_abs_exposure_nav"],
            "max_margin_requirement_jpy_at_1x": 200000.0 * exposure["max_reconstructed_gross_exposure_nav"],
            "max_inventory_age_seconds": max_age,
        },
        "arms": {
            arm: {
                "equity_multiple": multiples[arm],
                "log_growth": daily_metrics[arm]["log_growth"],
                "gross_edge_bps": decomposition_payload[arm]["edge_per_episode_bps"],
                "realized_cost_bps": decomposition_payload[arm]["cost_per_episode_bps"],
                "net_edge_bps": (
                    decomposition_payload[arm]["edge_per_episode_bps"]
                    - decomposition_payload[arm]["cost_per_episode_bps"]
                ),
                "cost_to_expected_move_ratio": (
                    decomposition_payload[arm]["cost_per_episode_bps"]
                    / decomposition_payload[arm]["edge_per_episode_bps"]
                    if decomposition_payload[arm]["edge_per_episode_bps"] > 0 else None
                ),
                "daily_mark_max_drawdown": daily_metrics[arm]["daily_mark_max_drawdown"],
                "conservative_equal_weight_component_intraday_dd_bound": -statistics.fmean(
                    abs(results[cycle]["periods"][month][arm]["max_drawdown"])
                    for cycle in ("V38", "V40")
                ),
                "terminal_inventory_mtm": 0.0,
                "terminal_open_inventory": 0,
            } for arm in ARMS
        },
        "decomposition": decomposition_payload,
        "allocation": "FIXED_EQUAL_COMPONENT_SLEEVES",
        "cycle_id": "V38_V40_FIXED_EQUAL_SLEEVE",
    }


def summarize_full_months(month_payloads: dict[str, dict]) -> dict:
    summary = {}
    for arm in ARMS:
        logs = [month_payloads[month]["arms"][arm]["log_growth"] for month in MONTHS]
        multiples = [month_payloads[month]["arms"][arm]["equity_multiple"] for month in MONTHS]
        mean_log = statistics.fmean(logs)
        summary[arm] = {
            "month_log_growth": dict(zip(MONTHS, logs)),
            "month_multiples": dict(zip(MONTHS, multiples)),
            "two_full_month_product_multiple": math.prod(multiples),
            "mean_full_month_log_growth": mean_log,
            "calendar_day_weighted_month_equivalent_log_growth": (
                sum(logs) / 61.0 * 30.5
            ),
            "target_monthly_log_growth": TARGET_LOG_GROWTH,
            "target_to_observed_log_growth_ratio": TARGET_LOG_GROWTH / mean_log if mean_log > 0 else None,
            "log_growth_gap": TARGET_LOG_GROWTH - mean_log,
            "worst_full_month_multiple": min(multiples),
            "all_full_months_at_least_2x": all(value >= 2.0 for value in multiples),
        }
    return summary


def backsolve(entity: str, month_payloads: dict[str, dict], summary: dict,
              current_workers: int) -> dict:
    output = {}
    for arm in ("EXECUTABLE_BASE", "ADVERSE_STRESS"):
        mean_log = summary[arm]["mean_full_month_log_growth"]
        if mean_log <= 0:
            output[arm] = {
                "status": "IMPOSSIBLE_AT_CURRENT_NEGATIVE_OR_ZERO_GROWTH_SIGN",
                "diagnostic_only": True,
            }
            continue
        factor = TARGET_LOG_GROWTH / mean_log
        mean_clusters = statistics.fmean(
            month_payloads[month]["N_eff"]["currency_time_clusters"] for month in MONTHS
        )
        mean_effective_risk = statistics.fmean(
            month_payloads[month]["decomposition"][arm]["effective_risk_nav_per_independent_cluster"]
            for month in MONTHS
        )
        mean_turnover = statistics.fmean(
            month_payloads[month]["trade_density"]["turnover_nav"] for month in MONTHS
        )
        current_net_edge = statistics.fmean(
            month_payloads[month]["decomposition"][arm]["edge_per_episode_bps"]
            - month_payloads[month]["decomposition"][arm]["cost_per_episode_bps"]
            for month in MONTHS
        )
        max_gross = max(month_payloads[month]["risk"]["max_gross_exposure_nav"] for month in MONTHS)
        max_currency = max(
            month_payloads[month]["risk"]["max_currency_abs_exposure_nav"] for month in MONTHS
        )
        drawdowns = []
        for month in MONTHS:
            arm_metrics = month_payloads[month]["arms"][arm]
            value = arm_metrics["max_drawdown"] if "max_drawdown" in arm_metrics else arm_metrics[
                "conservative_equal_weight_component_intraday_dd_bound"
            ]
            drawdowns.append(abs(value))
        max_dd = max(drawdowns)
        required_gross = max_gross * factor
        required_margin = 200000.0 * required_gross
        required_bets = mean_clusters * factor
        required_dd = max_dd * factor
        contradictions = []
        if required_gross > 1.0:
            contradictions.append("GROSS_CAP_1X")
        if max_currency * factor > 1.0:
            contradictions.append("CURRENCY_CAP_1X")
        if required_margin > 200000.0:
            contradictions.append("MARGIN_EXCEEDS_200000_JPY_AT_1X")
        if required_dd >= 1.0:
            contradictions.append("LINEAR_DD_EXHAUSTS_CAPITAL")
        if required_bets > statistics.fmean([31.0, 30.0]) * 7.0:
            contradictions.append("EXCEEDS_SEVEN_PAIR_DAILY_SIGNAL_TOPOLOGY")
        contradictions.append("LINEAR_SCALING_WOULD_SCALE_COST_AND_IS_NOT_ADMISSION_EVIDENCE")
        output[arm] = {
            "entity": entity,
            "diagnostic_only": True,
            "may_not_admit_strategy": True,
            "current_mean_full_month_log_growth": mean_log,
            "target_log_growth": TARGET_LOG_GROWTH,
            "required_growth_multiplier": factor,
            "current_currency_time_clusters_per_month": mean_clusters,
            "required_currency_time_clusters_linear": required_bets,
            "current_net_edge_per_episode_bps": current_net_edge,
            "required_net_edge_per_episode_bps_holding_N_eff_and_risk": (
                TARGET_LOG_GROWTH / (mean_clusters * mean_effective_risk) * 10000.0
            ),
            "current_effective_risk_nav_per_cluster": mean_effective_risk,
            "required_capital_rotation_turnover_nav_linear": mean_turnover * factor,
            "required_risk_multiplier_linear": factor,
            "required_gross_exposure_nav_linear": required_gross,
            "required_currency_exposure_nav_linear": max_currency * factor,
            "required_margin_jpy_at_1x_linear": required_margin,
            "required_max_drawdown_abs_linear": required_dd,
            "current_independent_worker_count": current_workers,
            "required_uncapped_full_cap_worker_equivalents": math.ceil(current_workers * factor),
            "contradictions": contradictions,
        }
    return output


def graph_identifiability() -> dict:
    pairs = sorted(component_v1.engine.UNIVERSE)
    nodes = sorted({currency for pair in pairs for currency in pair.split("_")})
    adjacency = {node: set() for node in nodes}
    for pair in pairs:
        left, right = pair.split("_")
        adjacency[left].add(right)
        adjacency[right].add(left)
    components = 0
    unseen = set(nodes)
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current] & unseen:
                unseen.remove(neighbor)
                stack.append(neighbor)
    edges = len(pairs)
    rank = len(nodes) - components
    cycle_space = edges - len(nodes) + components
    residual_df = edges - rank
    all_usd_crosses = all("USD" in pair.split("_") for pair in pairs)
    return {
        "pairs": pairs,
        "nodes": nodes,
        "node_count": len(nodes),
        "edge_count": edges,
        "connected_components": components,
        "oriented_incidence_rank": rank,
        "cycle_space_dimension": cycle_space,
        "overidentifying_residual_degrees_of_freedom": residual_df,
        "all_pairs_are_usd_crosses": all_usd_crosses,
        "is_tree": components == 1 and edges == len(nodes) - 1,
        "candidate_1_status": "REJECTED_STRUCTURALLY_WITHOUT_RETURN_OR_COST_DATA"
        if cycle_space == 0 or residual_df == 0 else "STRUCTURALLY_IDENTIFIABLE",
        "profit_or_cost_outcomes_used_for_rejection": False,
        "unregularized_pair_residual_direction_edge_allowed": False,
        "reconsideration_requires": "SEPARATE_COMPLETED_BID_ASK_CROSS_PAIR_DATA_EXPERIMENT",
        "allowed_current_star_observables": [
            "QUOTE_ORIENTATION_NORMALIZED_USD_COMMON_FACTOR_IMPULSE",
            "SIGN_BREADTH",
            "CROSS_SECTIONAL_DISPERSION_MAD",
            "LAMBDA_MAX_OVER_TRACE",
        ],
        "allowed_observable_role": "REGIME_INVENTORY_CAP_OR_FREEZE_NOT_DIRECTION_EDGE",
        "reason": (
            "SEVEN_USD_MAJOR_EDGES_FORM_AN_EIGHT_NODE_TREE_WITH_NO_REDUNDANT_CYCLE;_"
            "NODE_RETURNS_EXACTLY_FIT_EVERY_EDGE_AND_LEAVE_ZERO_GRAPH_RESIDUAL_DF"
        ),
    }


def pair_readback_summary(
    pair_rows: list[dict], aud_crosses: list[dict], derived_payload: dict
) -> dict:
    summary = {}
    for pair in SELECTED_PAIR_ARMS:
        official = [row for row in pair_rows if row["pair"] == pair and row["period"] == "WALK_FORWARD"]
        episodes = sum(
            row["N_eff"]["observed_nonoverlapping_pair_episodes"] for row in official
        )
        summary[pair] = {
            "sealed_cycles_read_back": [row["cycle_id"] for row in official],
            "sealed_cycle_count": len(official),
            "walk_forward_observed_pair_episodes_across_all_seals_not_independent": episodes,
            "classification": "TESTED_IN_SEALED_WALK_FORWARD_BUT_PREVIOUSLY_AGGREGATED"
            if episodes else "NO_REALIZED_EPISODE_IN_SEALED_WALK_FORWARD",
            "selection_for_next_matrix": "PAIR_PRIORITY_FIXED_FROM_CAUSE_AUDIT_NOT_WALK_FORWARD_RANKING",
            "autocorrelation_or_shared_usd_adjusted_n_eff": None,
        }
    return {
        "fixed_primary_pair_arms": list(SELECTED_PAIR_ARMS),
        "primary_pair_status": summary,
        "independent_pair_specialized_strategy_cycle_count": 0,
        "valid_sealed_cycles": derived_payload["deduplication"]["valid_sealed_cycles"],
        "valid_sealed_cycle_count": derived_payload["deduplication"]["valid_sealed_cycle_count"],
        "unique_raw_signal_stream_count": derived_payload["deduplication"][
            "unique_signal_id_set_count"
        ],
        "pair_signal_id_dedupe": derived_payload["deduplication"]["pair_signal_id_dedupe"],
        "v25_v27_direct_checks": derived_payload["direct_pair_checks"],
        "invalid_v34": next(
            item for item in derived_payload["failed_and_invalid_cycles"]
            if item["cycle_id"] == "V34"
        ),
        "pair_ranking_by_walk_forward_performed": False,
        "pair_selection_period": "TUNING_ONLY_THEN_FREEZE",
        "all_seven_frozen_pairs_have_direct_rows": len({row["pair"] for row in pair_rows}) == 7,
        "aud_related_crosses_present": bool(aud_crosses),
        "aud_related_cross_pairs": [row["pair"] for row in aud_crosses],
        "aud_cross_action": "DEFER_TO_SEPARATE_FAMILY_NO_CURRENT_RESULT_MERGE",
    }


def accounting_diagnostic(policy: dict) -> dict:
    source = policy["legacy_accounting_read_only_diagnostic"]
    cycles = {}
    for cycle, values in source["cycles"].items():
        cycles[cycle] = {
            **values,
            "base_delta_corrected_minus_sealed": (
                values["corrected_EXECUTABLE_BASE"] - values["sealed_EXECUTABLE_BASE"]
            ),
            "adverse_delta_corrected_minus_sealed": (
                values["corrected_ADVERSE_STRESS"] - values["sealed_ADVERSE_STRESS"]
            ),
            "corrected_both_below_2x": (
                values["corrected_EXECUTABLE_BASE"] < 2.0
                and values["corrected_ADVERSE_STRESS"] < 2.0
            ),
        }
    return {
        "classification": source["classification"],
        "method": source["method"],
        "sealed_signal_and_action_stream_changed": False,
        "cycles": cycles,
        "correction_reveals_hidden_2x": False,
        "reusable_as_official_seal": False,
        "next_required_action": source["future_requirement"],
    }


def legacy_actual_llm_readback(root: Path) -> dict[str, Any]:
    """Read historical actual-LLM evidence without treating it as a current seal."""
    v253_paths = {
        "contract": LEGACY_LLM_V253_ROOT / "contract_v253.json",
        "report": LEGACY_LLM_V253_ROOT / "report_v253_001.json",
        "runner": LEGACY_LLM_V253_ROOT / "run_walk_ab_v253.py",
        "test": LEGACY_LLM_V253_ROOT / "test_v253.py",
        "validation_report": LEGACY_LLM_V252_ROOT / "report_v252_001.json",
        "prompt": LEGACY_LLM_V252_ROOT / "actual_llm_prompt_v252_001.json",
        "decision": LEGACY_LLM_V252_ROOT / "actual_llm_decision_v252_001.json",
        "source_ledger": LEGACY_LLM_V250_ROOT / "decision_ledger_v250_001.jsonl.gz",
    }
    if any(not path.is_file() for path in v253_paths.values()):
        raise CauseFeasibilityError("legacy V253 actual-LLM artifact is incomplete")
    v253_before = {name: sha256_file(path) for name, path in v253_paths.items()}
    v253_contract = load_json(v253_paths["contract"])
    v253_report = load_json(v253_paths["report"])
    v252_report = load_json(v253_paths["validation_report"])
    v253_decision = load_json(v253_paths["decision"])
    if v253_report.get("result_sha256") != embedded_hash(v253_report, "result_sha256"):
        raise CauseFeasibilityError("legacy V253 result embedded hash mismatch")
    if v253_report.get("contract_sha256") != v253_before["contract"] \
            or v253_report.get("validation_report_sha256") != v253_before["validation_report"] \
            or v253_report.get("actual_llm_prompt_sha256") != v253_before["prompt"] \
            or v253_report.get("actual_llm_decision_sha256") != v253_before["decision"] \
            or v253_report.get("source_ledger_sha256") != v253_before["source_ledger"]:
        raise CauseFeasibilityError("legacy V253 source provenance hash mismatch")
    if v253_contract.get("same_proposal_stream") is not True \
            or v253_contract.get("same_execution_and_hard_guards") is not True \
            or v253_contract.get("reserved_holdout_opened") is not False \
            or v253_report.get("reserved_holdout_opened") is not False:
        raise CauseFeasibilityError("legacy V253 comparison or holdout boundary changed")
    bot_v253 = v253_report["results"]["bot"]
    llm_v253 = v253_report["results"]["actual_llm"]
    if bot_v253["configuration"]["weights"] != llm_v253["configuration"]["weights"] \
            or bot_v253["configuration"]["gross_cap"] != 1.0 \
            or llm_v253["configuration"]["gross_cap"] != 12.0 \
            or sorted(v253_report["results"]) != ["actual_llm", "bot"]:
        raise CauseFeasibilityError("legacy V253 leverage-confounded arm structure changed")
    expected_v253 = {
        "normal": (1.093389101612891, -0.0666401415320913, 0.9645509002239699),
        "adverse": (1.0379005274152473, -0.07739535779858808, 0.9591929992782117),
    }
    for arm, (multiple, drawdown, worst_month) in expected_v253.items():
        actual = llm_v253[arm]
        if not math.isclose(actual["total_multiple"], multiple, rel_tol=0.0, abs_tol=1e-15) \
                or not math.isclose(actual["max_drawdown"], drawdown,
                                    rel_tol=0.0, abs_tol=1e-15) \
                or not math.isclose(actual["minimum_monthly_multiple"], worst_month,
                                    rel_tol=0.0, abs_tol=1e-15) \
                or actual["months_at_or_above_2x"] != 0 \
                or actual["margin_closeouts"] != 0 \
                or actual["forced_terminal_or_margin_positions"] != 0:
            raise CauseFeasibilityError(f"legacy V253 actual-LLM metric changed: {arm}")
    if v253_decision.get("policy") != "H8_ONLY" \
            or v253_decision.get("mode") != "ADD_WITHIN_CAP" \
            or v253_decision.get("gross_cap") != 12 \
            or v253_decision.get("development_walk_seen") is not False \
            or v253_decision.get("reserved_holdout_seen") is not False \
            or v252_report.get("deterministic_bot_selection") != "H8_ONLY:L1" \
            or "H8_ONLY:L12" not in v252_report.get("grid", {}):
        raise CauseFeasibilityError("legacy V253 frozen decision provenance changed")
    v253_aud = []
    with gzip.open(v253_paths["source_ledger"], "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] == "development_walk" \
                    and row["candidate"] == "M15_H8:ridge:P0.0" \
                    and row["pair"] == "AUD_USD":
                v253_aud.append(row)
    if len(v253_aud) != 8 \
            or not math.isclose(sum(row["direction_correct"] for row in v253_aud) / 8.0,
                                0.875, rel_tol=0.0, abs_tol=1e-15) \
            or not math.isclose(sum(row["normal_return"] for row in v253_aud),
                                0.017594536077755626, rel_tol=0.0, abs_tol=1e-15) \
            or not math.isclose(sum(row["adverse_return"] for row in v253_aud),
                                0.01565322281193657, rel_tol=0.0, abs_tol=1e-15):
        raise CauseFeasibilityError("legacy V253 AUDUSD small-sample evidence changed")

    v15_names = (
        "preregister_v15.json", "prompt_packet_v15.json", "llm_portfolio_responses_v15.json",
        "portfolio_veto_v15.py", "ledger_v15.jsonl", "result_v15.json",
    )
    if not LEGACY_LLM_V15_ROOT.is_dir() \
            or any(not (LEGACY_LLM_V15_ROOT / name).is_file() for name in v15_names):
        raise CauseFeasibilityError("legacy V15 actual-LLM artifact is incomplete")
    v15_before = {
        name: sha256_file(LEGACY_LLM_V15_ROOT / name) for name in v15_names
    }
    v15_result = load_json(LEGACY_LLM_V15_ROOT / "result_v15.json")
    v15_prereg = load_json(LEGACY_LLM_V15_ROOT / "preregister_v15.json")
    v15_responses = load_json(LEGACY_LLM_V15_ROOT / "llm_portfolio_responses_v15.json")
    v15_rows = load_jsonl(LEGACY_LLM_V15_ROOT / "ledger_v15.jsonl")
    provenance = v15_result["provenance"]
    if provenance["prompts_sha256"] != v15_before["prompt_packet_v15.json"] \
            or provenance["responses_sha256"] != v15_before["llm_portfolio_responses_v15.json"] \
            or provenance["scorer_sha256"] != v15_before["portfolio_veto_v15.py"]:
        raise CauseFeasibilityError("legacy V15 prompt/response/scorer hash mismatch")
    chain = "0" * 64
    for sequence, row in enumerate(v15_rows, 1):
        if row["sequence"] != sequence or row["previous_hash"] != chain:
            raise CauseFeasibilityError("legacy V15 append-only ledger chain mismatch")
        unsigned = dict(row)
        record_hash = unsigned.pop("record_hash")
        chain = hashlib.sha256(json.dumps(
            unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if chain != record_hash:
            raise CauseFeasibilityError("legacy V15 record hash mismatch")
    if chain != provenance["ledger_terminal_hash"] \
            or len(v15_rows) != len(v15_responses["responses"]):
        raise CauseFeasibilityError("legacy V15 terminal ledger/proposal count mismatch")
    v15_normal_fills = [
        trade for row in v15_rows for trade in row["normal"]["LLM_PORTFOLIO_VETO"]
        if trade["status"] == "FILLED"
    ]
    v15_stress_fills = [
        trade for row in v15_rows for trade in row["stress_3x"]["LLM_PORTFOLIO_VETO"]
        if trade["status"] == "FILLED"
    ]
    normal_pnl = sum(item["net_pnl_quote"] for item in v15_normal_fills)
    stress_pnl = sum(item["net_pnl_quote"] for item in v15_stress_fills)
    result_arm = v15_result["arms"]["LLM_PORTFOLIO_VETO"]
    if len(v15_normal_fills) != result_arm["normal"]["filled_trades"] \
            or not math.isclose(normal_pnl, result_arm["normal"]["net_pnl_jpy"],
                                rel_tol=0.0, abs_tol=1e-12) \
            or not math.isclose(stress_pnl, result_arm["stress_3x"]["net_pnl_jpy"],
                                rel_tol=0.0, abs_tol=1e-12):
        raise CauseFeasibilityError("legacy V15 ledger/result metric mismatch")
    audjpy = [
        item for item in v15_normal_fills
        if item["decision_id"].startswith("AUDJPY") and item["expected_order"]["side"] == "SHORT"
    ]
    audjpy_stress = [
        item for item in v15_stress_fills
        if item["decision_id"] == audjpy[0]["decision_id"]
    ] if len(audjpy) == 1 else []
    if len(audjpy) != 1 or len(audjpy_stress) != 1 \
            or audjpy[0]["reason"] != "END_OF_DATA" \
            or audjpy_stress[0]["reason"] != "END_OF_DATA" \
            or not math.isclose(audjpy[0]["net_pnl_quote"], 67.27121008299028,
                                rel_tol=0.0, abs_tol=1e-12) \
            or not math.isclose(audjpy_stress[0]["net_pnl_quote"], 35.813630746988245,
                                rel_tol=0.0, abs_tol=1e-12):
        raise CauseFeasibilityError("legacy V15 AUDJPY evidence changed")

    v18_paths = {
        "result": LEGACY_LLM_V18_ROOT / "result_v18.json",
        "ledger": LEGACY_LLM_V18_ROOT / "audit_ledger_v18.jsonl",
    }
    if any(not path.is_file() for path in v18_paths.values()):
        raise CauseFeasibilityError("legacy V18 actual-LLM artifact is incomplete")
    v18_before = {name: sha256_file(path) for name, path in v18_paths.items()}
    v18_result = load_json(v18_paths["result"])
    v18_rows = load_jsonl(v18_paths["ledger"])
    v18_audjpy = [
        row for row in v18_rows
        if row["arm"] == "HYBRID_FINAL"
        and row["normal_outcome"]["status"] == "FILLED"
        and row["normal_outcome"]["decision_id"].startswith("AUDJPY")
    ]
    v18_arm = v18_result["arms"]["HYBRID_FINAL"]
    if len(v18_audjpy) != 1 \
            or v18_audjpy[0]["expected_order"]["side"] != "LONG" \
            or v18_audjpy[0]["normal_outcome"]["reason"] != "END_OF_DATA" \
            or not math.isclose(v18_arm["normal"]["net_pnl_jpy"], -65.76918005002624,
                                rel_tol=0.0, abs_tol=1e-12) \
            or not math.isclose(v18_arm["stress_3x"]["net_pnl_jpy"], -97.30754045001333,
                                rel_tol=0.0, abs_tol=1e-12) \
            or v18_result.get("profitability_claim_allowed") is not False \
            or v18_result.get("live_permission_allowed") is not False:
        raise CauseFeasibilityError("legacy V18 AUDJPY counterevidence changed")

    v13_result_path = root / PROTECTED_PATHS["v13_result"]
    v13_ledger_path = root / PROTECTED_PATHS["v13_ledger"]
    v13_result = load_json(v13_result_path)
    v13_rows = load_jsonl(v13_ledger_path)
    if v13_result.get("result_sha256") != embedded_hash(v13_result, "result_sha256") \
            or v13_result["proposal_ledger_sha256"] != sha256_file(v13_ledger_path):
        raise CauseFeasibilityError("V13 actual-LLM result/ledger hash mismatch")
    if v13_result.get("terminal_open_inventory") != 0 \
            or v13_result.get("terminal_inventory_mtm_hidden") is not False \
            or v13_result.get("external_orders") != 0 \
            or v13_result.get("live_authority") is not False:
        raise CauseFeasibilityError("V13 actual-LLM boundary changed")
    contained = [
        row for row in v13_rows
        if "2026-05-01" <= row["fill_time"][:10] < "2026-07-01"
        and row["exit_time"][:10] < "2026-07-01"
    ]
    if len(contained) != 370:
        raise CauseFeasibilityError("V13 contained walk-forward cohort changed")
    v13_additive = {
        arm: sum(row["scores"][arm]["net_return"] for row in contained) for arm in ARMS
    }
    for arm in ARMS:
        sealed = v13_result["periods"]["WALK_FORWARD"]["arms"][arm]["additive_return"]
        if not math.isclose(v13_additive[arm], sealed, rel_tol=0.0, abs_tol=1e-15):
            raise CauseFeasibilityError(f"V13 contained additive return changed: {arm}")
    aud_fill_cohort = [
        row for row in v13_rows
        if row["pair"] == "AUD_USD"
        and "2026-05-01" <= row["fill_time"][:10] < "2026-07-01"
    ]
    aud_contained = [row for row in contained if row["pair"] == "AUD_USD"]
    if len(aud_fill_cohort) != 56 or len(aud_contained) != 55:
        raise CauseFeasibilityError("V13 AUDUSD cohort counts changed")
    aud_fill_additive = {
        arm: sum(row["scores"][arm]["net_return"] for row in aud_fill_cohort) for arm in ARMS
    }

    ai_test_paths = {
        "result": AI_TEST_BOT_ROOT / "data/ai_test_bot_backtest.json",
        "implementation": AI_TEST_BOT_ROOT / "src/quant_rabbit/ai_test_bot.py",
    }
    if any(not path.is_file() for path in ai_test_paths.values()):
        raise CauseFeasibilityError("legacy AI Test Bot evidence is incomplete")
    ai_test_before = {name: sha256_file(path) for name, path in ai_test_paths.items()}
    ai_test = load_json(ai_test_paths["result"])
    ai_test_source = ai_test_paths["implementation"].read_text(encoding="utf-8")
    if "never calls a model API" not in ai_test_source \
            or not math.isclose(ai_test["summary"]["total_managed_net_jpy"], 40066.0732,
                                rel_tol=0.0, abs_tol=1e-9) \
            or ai_test.get("live_permission") is not False:
        raise CauseFeasibilityError("legacy AI Test Bot attribution evidence changed")

    if not BROKER_COHORT_REPORT.is_file():
        raise CauseFeasibilityError("legacy broker cohort evidence is missing")
    broker_before = sha256_file(BROKER_COHORT_REPORT)
    broker_report = load_json(BROKER_COHORT_REPORT)
    broker_windows = {item["id"]: item for item in broker_report["windows"]}
    broker_64d = broker_windows.get("QUADRUPLE_64D", {}).get("ALL_TRADES", {})
    if broker_report.get("overall_decision") != "REJECT" \
            or broker_report.get("prediction_rows") != [] \
            or broker_64d.get("trades_available") != 101 \
            or broker_64d.get("trades_selected") != 101 \
            or not math.isclose(broker_64d.get("net_jpy", math.nan), 15144.4802,
                                rel_tol=0.0, abs_tol=1e-9):
        raise CauseFeasibilityError("legacy broker cohort attribution evidence changed")

    current_registry = load_json(root / derived_pair.REGISTRY_PATH)
    current_cycles = {item["cycle_id"]: item for item in current_registry["cycles"]}
    no_actual_llm_cycles = [f"V{ordinal}" for ordinal in range(26, 42)]
    if current_cycles["V25"]["proposal_provenance"]["model_identity"] \
            != "LEGACY_TASK_NOT_MACHINE_VERIFIED" \
            or any(
                current_cycles[cycle]["proposal_provenance"]["model_identity"]
                != "NO_ACTUAL_LLM_USED"
                for cycle in no_actual_llm_cycles
            ):
        raise CauseFeasibilityError("current V25-V41 LLM provenance classification changed")

    v253_after = {name: sha256_file(path) for name, path in v253_paths.items()}
    v15_after = {name: sha256_file(LEGACY_LLM_V15_ROOT / name) for name in v15_names}
    v18_after = {name: sha256_file(path) for name, path in v18_paths.items()}
    ai_test_after = {name: sha256_file(path) for name, path in ai_test_paths.items()}
    broker_after = sha256_file(BROKER_COHORT_REPORT)
    if v253_before != v253_after or v15_before != v15_after \
            or v18_before != v18_after or ai_test_before != ai_test_after \
            or broker_before != broker_after:
        raise CauseFeasibilityError("legacy source changed during read-only actual-LLM audit")
    return {
        "classification": "LEGACY_ACTUAL_LLM_EVIDENCE_NOT_CURRENT_STRATEGY_SEAL",
        "v253_development_walk_inventory_policy": {
            "source_root": str(LEGACY_LLM_V253_ROOT),
            "file_hashes": v253_before,
            "development_walk": v253_contract["development_walk"],
            "proposal_stream_shared": True,
            "execution_and_hard_guards_shared": True,
            "bot_configuration": bot_v253["configuration"],
            "actual_llm_configuration": llm_v253["configuration"],
            "bot": {"normal": bot_v253["normal"], "adverse": bot_v253["adverse"]},
            "actual_llm": {"normal": llm_v253["normal"], "adverse": llm_v253["adverse"]},
            "same_policy_weights": True,
            "same_gross_cap": False,
            "bot_gross_cap": 1.0,
            "actual_llm_gross_cap": 12.0,
            "same_cap_bot_control_present": False,
            "same_cap_mechanical_validation_control_present": False,
            "validation_grid_contains_h8_cap12": True,
            "frozen_actual_llm_decision": {
                "policy": v253_decision["policy"],
                "mode": v253_decision["mode"],
                "gross_cap": v253_decision["gross_cap"],
                "development_walk_seen": v253_decision["development_walk_seen"],
                "reserved_holdout_seen": v253_decision["reserved_holdout_seen"],
            },
            "aud_usd_development_diagnostic": {
                "episodes": len(v253_aud),
                "direction_accuracy": sum(row["direction_correct"] for row in v253_aud)
                / len(v253_aud),
                "normal_additive_return": sum(row["normal_return"] for row in v253_aud),
                "adverse_additive_return": sum(row["adverse_return"] for row in v253_aud),
                "small_sample": True,
                "llm_selected_individual_directions": False,
            },
            "holdout": "UNOPENED",
            "adopted": False,
            "interpretation": "STRONGEST_REPRODUCIBLE_LEGACY_LLM_RESULT_BUT_LEVERAGE_CONFOUNDED",
            "incremental_llm_edge_identified": False,
        },
        "v15_jpy_portfolio_veto": {
            "source_root": str(LEGACY_LLM_V15_ROOT),
            "file_hashes": v15_before,
            "ledger_chain_verified": True,
            "decision_groups": len(v15_rows),
            "model": v15_responses["model"],
            "model_version": v15_responses["model_version"],
            "llm_permission": v15_prereg["llm_permission"],
            "normal": result_arm["normal"],
            "stress_3x": result_arm["stress_3x"],
            "aud_jpy_short_normal_net_pnl_jpy": audjpy[0]["net_pnl_quote"],
            "aud_jpy_short_stress_net_pnl_jpy": audjpy_stress[0]["net_pnl_quote"],
            "aud_jpy_exit_reason": "END_OF_DATA",
            "aud_jpy_episode_count": 1,
            "single_episode_may_support_edge_claim": False,
            "holdout": "UNOPENED",
            "adopted": False,
            "interpretation": "SHORT_TWO_DAY_NORMAL_GAIN_BUT_STRESS_SIGN_REVERSAL_NOT_ADOPTION_EVIDENCE",
        },
        "v18_aud_jpy_counterexample": {
            "source_root": str(LEGACY_LLM_V18_ROOT),
            "file_hashes": v18_before,
            "arm": "HYBRID_FINAL",
            "side": "LONG",
            "episode_count": 1,
            "normal_net_pnl_jpy": v18_arm["normal"]["net_pnl_jpy"],
            "stress_3x_net_pnl_jpy": v18_arm["stress_3x"]["net_pnl_jpy"],
            "exit_reason": "END_OF_DATA",
            "profitability_claim_allowed": False,
            "interpretation": "ONE_SHORT_WIN_AND_ONE_LONG_LOSS_ARE_MUTUAL_COUNTEREVIDENCE_NOT_AUDJPY_EDGE",
        },
        "v13_usd_major_inventory_policy": {
            "result_path": PROTECTED_PATHS["v13_result"],
            "result_file_sha256": sha256_file(v13_result_path),
            "ledger_path": PROTECTED_PATHS["v13_ledger"],
            "ledger_file_sha256": sha256_file(v13_ledger_path),
            "contained_walk_forward_signals": len(contained),
            "contained_additive_returns": v13_additive,
            "raw_to_base_drag": v13_additive["RAW_SIGNAL"] - v13_additive["EXECUTABLE_BASE"],
            "raw_to_adverse_drag": v13_additive["RAW_SIGNAL"] - v13_additive["ADVERSE_STRESS"],
            "aud_usd_fill_date_cohort": {
                "signals": len(aud_fill_cohort),
                "additive_returns": aud_fill_additive,
                "official_contained_period_metric": False,
                "contained_episode_count": len(aud_contained),
            },
            "terminal_open_inventory": 0,
            "adopted": False,
            "interpretation": "RAW_EDGE_COLLAPSES_THROUGH_BASE_AND_ADVERSE_COST_MARGIN",
        },
        "ai_test_bot_deterministic_not_actual_llm": {
            "source_root": str(AI_TEST_BOT_ROOT),
            "file_hashes": ai_test_before,
            "managed_net_pnl_jpy": ai_test["summary"]["total_managed_net_jpy"],
            "validation_days": ai_test["summary"]["validation_days"],
            "selected_trades": ai_test["summary"]["selected_trades"],
            "implementation_contract_says_model_api_never_called": True,
            "mechanism": "TRAILING_PRIOR_BUCKET_SELECTION_PLUS_POST_OUTCOME_LOSS_CAP",
            "actual_llm_attribution_proven": False,
            "included_as_actual_llm_profit_evidence": False,
        },
        "broker_cohort_64d_101_trades": {
            "source_path": str(BROKER_COHORT_REPORT),
            "source_file_sha256": broker_before,
            "net_pnl_jpy": broker_64d["net_jpy"],
            "trades": broker_64d["trades_selected"],
            "overall_decision": broker_report["overall_decision"],
            "actual_llm_or_codex_attribution_proven": False,
            "skip_or_counterfactual_ledger_present": False,
            "thesis_ledger_present": False,
            "included_as_ai_profit_evidence": False,
        },
        "current_v25_v41_actual_llm_status": {
            "machine_verified_actual_llm_strategy_arm_present": False,
            "v25_proposal_identity": "LEGACY_TASK_NOT_MACHINE_VERIFIED",
            "v26_through_v41_registry_identity": "NO_ACTUAL_LLM_USED",
            "v26_through_v41_cycles_verified": no_actual_llm_cycles,
            "registry_file_sha256": sha256_file(root / derived_pair.REGISTRY_PATH),
            "interpretation": "CURRENT_SESSION_WORKERS_DO_NOT_TEST_INCREMENTAL_LLM_EFFECT",
        },
        "diagnostic_priority_after_bot_pair_repair_plumbing": [
            {
                "priority": 1,
                "diagnostic": "V253_LLM_INCREMENTAL_EDGE_SEPARATION",
                "fixed_gross_cap": 12.0,
                "arms": [
                    "BOT_FIXED_SAME_CAP",
                    "MECHANICAL_VALIDATION_RULE_SAME_CAP",
                    "FROZEN_ACTUAL_LLM_SAME_CAP",
                ],
                "same_proposal_stream_risk_cost_required": True,
                "holdout": "UNOPENED",
                "profit_search": False,
            },
            {
                "priority": 2,
                "diagnostic": "V15_STYLE_JPY_CROSS_VETO_AB",
                "precondition": "HASH_FIXED_M1_JPY_CROSS_BID_ASK_CORPUS_WITH_FULL_COMPARABLE_MONTHS",
                "arms": ["BOT_TOP_ONE", "FROZEN_ACTUAL_LLM_VETO"],
                "single_episode_aud_result_may_select_or_admit": False,
            },
            {
                "priority": 3,
                "diagnostic": "V13_COST_MARGIN_CAUSE_EVIDENCE_ONLY",
                "new_run_required": False,
            },
        ],
        "portable_hypotheses_not_adopted": [
            "V253_SAME_CAP_BOT_MECHANICAL_AND_ACTUAL_LLM_THREE_ARM_DIAGNOSTIC",
            "NINETEEN_DECISION_TIME_FIXED_CANDIDATE_OR_NONE_VETO",
            "AUD_JPY_SEPARATE_REAL_BID_ASK_FAMILY_IF_SOURCE_EXISTS",
            "STRUCTURED_INVENTORY_POLICY_ARM",
        ],
        "primary_inference": (
            "NORMAL_SMALL_GAINS_EXIST_AND_V253_IS_REPRODUCIBLE_BUT_INCREMENTAL_LLM_EDGE_"
            "IS_NOT_IDENTIFIED_BECAUSE_CAP_DIFFERS;_REPEATED_ADVERSE_SIGN_REVERSAL_POINTS_"
            "TO_GROSS_EDGE_TO_COST_MARGIN_AND_SHORT_HORIZON_EXECUTION_FRAGILITY"
        ),
        "profit_or_adoption_claim_allowed": False,
    }


def build(root: Path) -> dict[str, Any]:
    policy_path = root / POLICY_PATH
    policy = load_json(policy_path)
    if policy.get("classification") != "NON_STRATEGY_ORCHESTRATOR_CAUSE_FEASIBILITY_POLICY" \
            or policy.get("authority") != AUTHORITY:
        raise CauseFeasibilityError("policy identity or zero-authority boundary changed")
    if not math.isclose(policy["profit_gate"]["target_monthly_log_growth"], TARGET_LOG_GROWTH,
                        rel_tol=0.0, abs_tol=1e-15):
        raise CauseFeasibilityError("monthly 2x log target changed")
    future_gate = policy["future_geometric_mean_profit_gate"]
    incomplete_fields = (
        "near_target_tolerance", "evaluation_month_count", "worst_month_floor",
        "maximum_drawdown_guard", "margin_guard", "ruin_guard",
    )
    if any(future_gate[field] is not None for field in incomplete_fields) \
            or future_gate["may_register_or_execute_future_profit_gate"] is not False:
        raise CauseFeasibilityError("unspecified future geometric gate was guessed or enabled")
    example_g = math.exp(statistics.fmean([
        math.log(future_gate["example"]["first_month_multiple"]),
        math.log(future_gate["example"]["required_second_month_multiple_for_two_month_G_2"]),
    ]))
    if not math.isclose(example_g, 2.0, rel_tol=0.0, abs_tol=1e-15):
        raise CauseFeasibilityError("future geometric gate formula example changed")
    legacy_llm_contract = policy["legacy_actual_llm_evidence_contract"]
    if legacy_llm_contract["current_v25_v41_machine_verified_actual_llm_strategy_arm"] \
            is not False \
            or legacy_llm_contract["v253"]["same_cap_control_present"] is not False \
            or legacy_llm_contract["v253"]["profit_or_incremental_llm_edge_claim_allowed"] \
            is not False \
            or legacy_llm_contract["past_positive_jpy_may_not_be_imported_into_current_profit_gate"] \
            is not True:
        raise CauseFeasibilityError("legacy actual-LLM evidence policy overstates attribution")
    component_v2_payload = component_v2.validate(root)
    derived_payload = derived_pair.validate(root)
    before = protected_hashes(root)
    results, ledgers = {}, {}
    for cycle, spec in sorted(CANDIDATES.items()):
        results[cycle], ledgers[cycle] = verify_sealed_cycle(root, spec)
    source_provenance = derived_payload["source_readback"]
    pair_rows = load_jsonl(root / derived_pair.METRICS_PATH)
    reconciliation = derived_payload["portfolio_reconciliation"]
    aud_cross_rows = aud_cross_inventory()
    pair_path = root / PAIR_READBACK_PATH
    aud_path = root / AUD_CROSS_INVENTORY_PATH
    atomic_text(aud_path, "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                                  for row in aud_cross_rows))

    candidates_by_cycle = {
        item["cycle_id"]: item for item in load_json(root / component_v1.REGISTRY_PATH)["candidates"]
    }
    exposure_maps = {
        cycle: component_v2.exposure_map(root, candidates_by_cycle[cycle])
        for cycle in ("V38", "V40")
    }
    daily_rows = load_jsonl(root / component_v2.DAILY_PORTFOLIO_PATH)
    entities: dict[str, dict[str, dict]] = {"V38": {}, "V40": {},
                                           "V38_V40_FIXED_EQUAL_SLEEVE": {}}
    for month in MONTHS:
        start, end = PERIODS[month]
        for cycle in ("V38", "V40"):
            exposure = exposure_metrics(exposure_maps[cycle], start, end)
            entities[cycle][month] = component_month_payload(
                cycle, results[cycle], month, exposure
            )
        combo_exposure = combination_exposure_metrics(exposure_maps, start, end)
        entities["V38_V40_FIXED_EQUAL_SLEEVE"][month] = combination_month_payload(
            results, month, combo_exposure, daily_rows
        )
    summaries = {entity: summarize_full_months(months) for entity, months in entities.items()}
    backsolves = {
        entity: backsolve(entity, months, summaries[entity], 2 if entity.startswith("V38_V40") else 1)
        for entity, months in entities.items()
    }
    graph = graph_identifiability()
    if graph["candidate_1_status"] != "REJECTED_STRUCTURALLY_WITHOUT_RETURN_OR_COST_DATA":
        raise CauseFeasibilityError("candidate priority 1 did not fail its structural precondition")

    pair_summary = pair_readback_summary(pair_rows, aud_cross_rows, derived_payload)
    accounting = accounting_diagnostic(policy)
    legacy_llm = legacy_actual_llm_readback(root)
    envelope_contract = policy["next_feasibility_envelope_contract"]
    envelope_missing = (
        "adverse_cost_formula", "realistic_corrected_lcb_definition", "drawdown_guard",
        "cvar_guard", "margin_guard", "ruin_guard",
    )
    if any(envelope_contract[field] is not None for field in envelope_missing) \
            or envelope_contract["current_execution_authorized"] is not False:
        raise CauseFeasibilityError("feasibility envelope missing fields were guessed or enabled")
    range_feasibility = derived_payload["daily_range_and_oracle_feasibility"]
    normal_oracle_by_month = {
        month: values["oracle_capture_solutions"]
        for month, values in range_feasibility["completed_daily_range"].items()
    }
    feasibility_envelope = {
        **envelope_contract,
        "current_verified_normal_cost_oracle": range_feasibility["completed_daily_range"],
        "normal_oracle_cap_grid_by_month": normal_oracle_by_month,
        "current_1x_required_capture_percent_by_month": {
            month: values["1.0"]["required_daily_high_low_capture_percent"]
            for month, values in normal_oracle_by_month.items()
        },
        "perfect_full_range_normal_cost_cap20_ceiling_by_month": {
            month: values["20.0"]["perfect_full_range_normal_cost_ceiling_multiple"]
            for month, values in normal_oracle_by_month.items()
        },
        "current_1x_capacity_insufficient": range_feasibility[
            "current_1x_required_capture_exceeds_full_daily_range"
        ],
        "entry_or_exit_tuning_alone_can_close_current_1x_gap": False,
        "missing_before_executable": list(envelope_missing),
        "adverse_or_realistic_lcb_result_inferred": False,
        "gross_cap_change_authorized": False,
        "capacity_interpretation": (
            "CURRENT_FAMILY_UNIVERSE_OPPORTUNITY_SET_AT_1X_IS_CAPACITY_INSUFFICIENT;_"
            "THIS_IS_NOT_A_CLAIM_THAT_ALL_FX_IS_IMPOSSIBLE"
        ),
    }
    sign_review = component_v2_payload["sign_aware_pair_reviews"][0]
    v38_max_age = max(
        results["V38"]["periods"][month]["EXECUTABLE_BASE"]["max_inventory_age_seconds"]
        for month in MONTHS
    )
    failure_taxonomy = {
        "dominant": [
            "SIGNAL_EDGE_INSUFFICIENCY",
            "INDEPENDENT_OPPORTUNITY_DENSITY_INSUFFICIENCY",
            "TARGET_FEASIBILITY",
        ],
        "SIGNAL_EDGE_INSUFFICIENCY": {
            "status": "DOMINANT",
            "evidence": {
                entity: summaries[entity]["RAW_SIGNAL"]["mean_full_month_log_growth"]
                for entity in summaries
            },
            "v25_pair_direct_readback": derived_payload["direct_pair_checks"]["V25"],
            "interpretation": "RAW_GROWTH_IS_ORDERS_OF_MAGNITUDE_BELOW_LN2_BEFORE_COST",
        },
        "INDEPENDENT_OPPORTUNITY_DENSITY_INSUFFICIENCY": {
            "status": "DOMINANT",
            "evidence": {
                entity: statistics.fmean(
                    entities[entity][month]["N_eff"]["currency_time_clusters"] for month in MONTHS
                ) for entity in entities
            },
            "interpretation": "CURRENCY_TIME_CLUSTERS_NOT_TICKETS_BIND_THE_AVAILABLE_BETS",
        },
        "WORKER_CORRELATION_OR_DUPLICATION": {
            "status": "NOT_DOMINANT_BUT_REVIEWED",
            "daily_base_return_correlation": sign_review["daily_base_return_correlation"],
            "classification": sign_review["classification"],
            "adverse_cvar_improvement_fraction": sign_review["adverse_tail"]["cvar_improvement_fraction"],
            "interpretation": "NEGATIVE_CORRELATION_IMPROVES_TAIL_BUT_DOES_NOT_MULTIPLY_EXPECTED_GROWTH",
        },
        "COST_DRAG": {
            "status": "MATERIAL_SECONDARY",
            "base_and_adverse_gap": {
                entity: summaries[entity]["EXECUTABLE_BASE"]["mean_full_month_log_growth"]
                - summaries[entity]["ADVERSE_STRESS"]["mean_full_month_log_growth"]
                for entity in summaries
            },
            "pair_evidence": {
                "EUR_USD": "V25_RAW_AND_BASE_POSITIVE_BUT_ADVERSE_NEGATIVE",
                "USD_JPY": "V25_DIRECTION_ACCURACY_ABOVE_HALF_BUT_BASE_AND_ADVERSE_NEGATIVE",
                "AUD_USD": "V25_RAW_BASE_AND_ADVERSE_NEGATIVE_WITH_MONTHLY_REGIME_SIGN_CHANGE",
            },
            "actual_llm_primary_example": {
                "cycle": "V13",
                "signals": legacy_llm["v13_usd_major_inventory_policy"]
                ["contained_walk_forward_signals"],
                "additive_returns": legacy_llm["v13_usd_major_inventory_policy"]
                ["contained_additive_returns"],
                "interpretation": "RAW_POSITIVE_BASE_NEAR_ZERO_ADVERSE_NEGATIVE",
            },
            "interpretation": "COST_FLIPS_OR_WEAKENS_MONTHS_BUT_REMOVING_ALL_COST_STILL_DOES_NOT_REACH_2X",
        },
        "INVENTORY_DWELL": {
            "status": "MATERIAL_FOR_V38",
            "v38_nominal_ledger_exit_seconds": 14100,
            "v38_official_target_hold_seconds": CANDIDATES["V38"]["horizon_seconds"],
            "v38_observed_max_inventory_age_seconds": v38_max_age,
            "interpretation": "V38_POSITIVE_RESULT_IS_MULTI_DAY_CARRY_NOT_THE_NOMINAL_FOUR_HOUR_RAW_NARRATIVE",
        },
        "RISK_BUDGET": {
            "status": "BINDING",
            "gross_cap": 1.0,
            "currency_cap": 1.0,
            "post_hoc_leverage_allowed": False,
            "interpretation": "LINEAR_RISK_BACKSOLVES_EXCEED_CAP_AND_MARGIN",
        },
        "DATA_INFORMATION_LIMIT": {
            "status": "BINDING",
            "may_june_reused_development_not_pure_holdout": True,
            "untouched_holdout": "UNOPENED",
            "graph_cycle_space_dimension": graph["cycle_space_dimension"],
            "interpretation": "NO_GRAPH_RESIDUAL_DF_AND_NO_NEW_FULL_MONTH_UNTOUCHED_OUTER_SLICE",
        },
        "TARGET_FEASIBILITY": {
            "status": "CURRENT_1X_STRUCTURE_ORACLE_CEILING_BELOW_2X",
            "target_monthly_log_growth": TARGET_LOG_GROWTH,
            "normal_shortfall_factors": {
                entity: summaries[entity]["EXECUTABLE_BASE"]["target_to_observed_log_growth_ratio"]
                for entity in summaries
            },
            "adverse_shortfall_factors": {
                entity: summaries[entity]["ADVERSE_STRESS"]["target_to_observed_log_growth_ratio"]
                for entity in summaries
            },
            "completed_daily_range_resource_exists": True,
            "range_shortage_is_dominant": False,
            "oracle_feasibility": derived_payload["daily_range_and_oracle_feasibility"],
            "current_opportunity_set_at_1x_capacity_insufficient": True,
            "claim_that_all_fx_is_impossible": False,
            "perfect_normal_cost_oracle_cap20_reaches_2x_in_both_months": all(
                value > 2.0 for value in feasibility_envelope[
                    "perfect_full_range_normal_cost_cap20_ceiling_by_month"
                ].values()
            ),
            "perfect_adverse_or_realistic_lcb_cap20_known": False,
            "leverage_rescue_authorized": False,
            "interpretation": (
                "AT_AUTHORIZED_1X_THE_LOOKAHEAD_FULL_DAILY_HIGH_LOW_ORACLE_REQUIRES_"
                "MORE_THAN_100_PERCENT_CAPTURE_IN_BOTH_MONTHS_SO_CAUSAL_2X_IS_STRUCTURALLY_"
                "IMPOSSIBLE_UNDER_THE_ONE_ROUNDTRIP_TWO_PAIR_ASSUMPTION"
            ),
        },
    }
    selected_architecture = policy["family_priority_without_outcome_comparison"][1]
    selected_family = policy["pair_family_priority_without_outcome_comparison"][0]
    v42_decision = {
        "existing_v42_work_order_sha256": before["v42_work_order"],
        "existing_v42_work_order_changed": False,
        "dst_only_strategy_execution": "NO_GO",
        "dst_role": "CHRONOLOGY_FOUNDATION_ONLY_NOT_NEW_EDGE",
        "candidate_1": graph,
        "selected_candidate_priority": 2,
        "selected_family_id": selected_family["family_id"],
        "selected_pair": selected_family["pair"],
        "selected_architecture_id": selected_architecture["family_id"],
        "selection_used_return_outcome_comparison": False,
        "selection_reason": (
            "GRAPH_RESIDUAL_IS_STRUCTURALLY_UNIDENTIFIABLE_AND_DERIVED_PAIR_EVIDENCE_SHOWS_"
            "EUR_USD_SMALL_RAW_EDGE_LOSES_TO_ADVERSE_COST_SO_H4_TO_M15_HIERARCHY_TARGETS_"
            "GROSS_MOVE_TO_COST_AND_EDGE_TIMES_DENSITY"
        ),
        "candidate_3_status": "DEFERRED_NOT_OUTCOME_EVALUATED",
        "current_official_v42_execution_authorized": False,
        "v42_may_proceed_only_as": "REDESIGNED_BOT_ONLY_SCALP_INTRADAY_FAMILY_AFTER_JPY_ACCOUNTING_AND_PREREG",
        "pre_fixed_improvement_gates": {
            key: value for key, value in selected_family.items()
            if key.startswith("minimum_") or key.startswith("maximum_")
        },
        "dominant_shortage_success_contract": envelope_contract[
            "next_family_success_condition"
        ],
        "small_positive_result_disposition": "COMPONENT_CANDIDATE_ONLY_NOT_PROFIT_ADMISSION",
    }
    research_matrix = {
        "pair_specialization": {
            **policy["pair_specialization_matrix"],
            "direct_readback": pair_summary,
            "cause_dominant_first_pair": selected_family["pair"],
            "pair_family_priority": policy["pair_family_priority_without_outcome_comparison"],
            "pair_outcomes_may_not_be_net_hidden": True,
        },
        "two_horizon": policy["two_horizon_matrix"],
        "indicator_factory": policy["indicator_factory_matrix"],
        "historical_and_forward_shadow": policy["historical_and_forward_shadow_matrix"],
        "actual_llm_ab": {
            **policy["actual_llm_ab_matrix"],
            "legacy_direct_readback": legacy_llm,
            "priority_after_bot_pair_repair_plumbing": legacy_llm[
                "diagnostic_priority_after_bot_pair_repair_plumbing"
            ],
        },
        "sequencing": [
            "V42_BOT_ONLY_SCALP_INTRADAY_FAMILY_WITH_H4_CONTEXT",
            "SEPARATE_SWING_CYCLE_AFTER_INTRADAY_SEAL",
            "ACTUAL_LLM_AB_ONLY_AFTER_BOT_ONLY_SEAL",
            "ZERO_ORDER_FORWARD_SHADOW_ONLY_AFTER_FEED_AND_VERSION_ARE_EXPLICITLY_FROZEN",
            "AUD_CROSSES_ONLY_IN_SEPARATE_FAMILY",
        ],
    }
    after = protected_hashes(root)
    if before != after:
        raise CauseFeasibilityError("historical strategy artifact changed during read-only audit")
    payload = {
        "schema_version": 1,
        "audit_id": "FX_MONTHLY_2X_CAUSE_FEASIBILITY_AUDIT_V1",
        "classification": "NON_STRATEGY_READ_ONLY_CAUSE_FEASIBILITY_EVIDENCE",
        "policy_path": POLICY_PATH,
        "policy_sha256": sha256_file(policy_path),
        "target": policy["profit_gate"],
        "future_geometric_mean_gate": {
            **future_gate,
            "example_computed_geometric_mean": example_g,
            "executable": False,
            "blocking_unspecified_fields": list(incomplete_fields),
        },
        "evaluation_calendar": policy["evaluation_calendar"],
        "source_provenance": source_provenance,
        "derived_pair_audit_path": derived_pair.AUDIT_PATH,
        "derived_pair_audit_file_sha256": sha256_file(root / derived_pair.AUDIT_PATH),
        "derived_pair_audit_sha256": derived_payload["audit_sha256"],
        "sealed_pair_readback_path": PAIR_READBACK_PATH,
        "sealed_pair_readback_sha256": sha256_file(pair_path),
        "aud_cross_inventory_path": AUD_CROSS_INVENTORY_PATH,
        "aud_cross_inventory_sha256": sha256_file(aud_path),
        "pair_reconstruction_reconciliation": reconciliation,
        "entities": entities,
        "full_month_summaries": summaries,
        "diagnostic_backsolves": backsolves,
        "failure_taxonomy": failure_taxonomy,
        "pair_direct_readback": pair_summary,
        "completed_daily_range_and_oracle_feasibility": derived_payload[
            "daily_range_and_oracle_feasibility"
        ],
        "next_feasibility_envelope": feasibility_envelope,
        "graph_identifiability": graph,
        "legacy_accounting_read_only_diagnostic": accounting,
        "legacy_actual_llm_read_only_evidence": legacy_llm,
        "v42_go_no_go": v42_decision,
        "next_research_matrix": research_matrix,
        "component_sign_aware_registry_file_sha256": sha256_file(root / component_v2.REGISTRY_PATH),
        "protected_strategy_artifact_hashes": {"before": before, "after": after,
                                                   "unchanged": before == after},
        "holdout_state": "UNOPENED",
        "strategy_adoption_authorized": False,
        "profit_gate_pass_inferred": False,
        "official_strategy_run_performed": False,
        "external_orders": 0,
        "authority": AUTHORITY,
    }
    payload["audit_sha256"] = embedded_hash(payload, "audit_sha256")
    atomic_text(root / AUDIT_PATH,
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return payload


def validate(root: Path) -> dict[str, Any]:
    payload = load_json(root / AUDIT_PATH)
    if payload.get("audit_sha256") != embedded_hash(payload, "audit_sha256"):
        raise CauseFeasibilityError("audit embedded hash mismatch")
    policy_path = root / payload.get("policy_path", "")
    if not policy_path.is_file() or sha256_file(policy_path) != payload.get("policy_sha256"):
        raise CauseFeasibilityError("audit policy hash mismatch")
    if payload.get("authority") != AUTHORITY or payload.get("external_orders") != 0 \
            or payload.get("holdout_state") != "UNOPENED":
        raise CauseFeasibilityError("audit authority or holdout boundary changed")
    if payload.get("strategy_adoption_authorized") is not False \
            or payload.get("profit_gate_pass_inferred") is not False \
            or payload.get("official_strategy_run_performed") is not False:
        raise CauseFeasibilityError("diagnostic overstates strategy evidence")
    for path_field, hash_field in (
        ("sealed_pair_readback_path", "sealed_pair_readback_sha256"),
        ("aud_cross_inventory_path", "aud_cross_inventory_sha256"),
        ("derived_pair_audit_path", "derived_pair_audit_file_sha256"),
    ):
        artifact = root / payload[path_field]
        if not artifact.is_file() or sha256_file(artifact) != payload[hash_field]:
            raise CauseFeasibilityError(f"audit evidence hash mismatch: {path_field}")
    protected = payload.get("protected_strategy_artifact_hashes", {})
    current = protected_hashes(root)
    if protected.get("unchanged") is not True or protected.get("before") != current \
            or protected.get("after") != current:
        raise CauseFeasibilityError("protected V38/V40/V41/V42 evidence changed")
    if payload["graph_identifiability"]["cycle_space_dimension"] != 0 \
            or payload["graph_identifiability"]["overidentifying_residual_degrees_of_freedom"] != 0:
        raise CauseFeasibilityError("zero-residual graph gate changed")
    gate = payload["v42_go_no_go"]
    if gate["dst_only_strategy_execution"] != "NO_GO" \
            or gate["current_official_v42_execution_authorized"] is not False \
            or gate["selected_family_id"] != "EUR_USD_H4_REGIME_TO_M15_ENTRY_TIMING_HIERARCHICAL":
        raise CauseFeasibilityError("V42 cause gate changed")
    future = payload["future_geometric_mean_gate"]
    if future["executable"] is not False or future["may_register_or_execute_future_profit_gate"] is not False \
            or not math.isclose(future["example_computed_geometric_mean"], 2.0,
                                rel_tol=0.0, abs_tol=1e-15):
        raise CauseFeasibilityError("future geometric gate incompleteness changed")
    oracle = payload["completed_daily_range_and_oracle_feasibility"]
    if oracle["current_1x_required_capture_exceeds_full_daily_range"] is not True \
            or oracle["may_admit_strategy"] is not False:
        raise CauseFeasibilityError("oracle feasibility classification changed")
    envelope = payload["next_feasibility_envelope"]
    if envelope["fixed_gross_cap_grid"] != [1.0, 4.0, 8.0, 12.0, 20.0] \
            or envelope["current_execution_authorized"] is not False \
            or envelope["gross_cap_change_authorized"] is not False \
            or envelope["adverse_or_realistic_lcb_result_inferred"] is not False \
            or envelope["current_1x_capacity_insufficient"] is not True \
            or envelope["entry_or_exit_tuning_alone_can_close_current_1x_gap"] is not False:
        raise CauseFeasibilityError("feasibility envelope authorization changed")
    expected_capture = derived_pair.EXPECTED_ORACLE_CAPTURE_PERCENT
    for month, caps in expected_capture.items():
        actual_caps = envelope["normal_oracle_cap_grid_by_month"][month]
        for cap, expected_percent in caps.items():
            if not math.isclose(
                actual_caps[cap]["required_daily_high_low_capture_percent"],
                expected_percent, rel_tol=0.0, abs_tol=1e-10,
            ):
                raise CauseFeasibilityError("feasibility envelope oracle grid changed")
    legacy_llm = payload["legacy_actual_llm_read_only_evidence"]
    if legacy_llm != legacy_actual_llm_readback(root):
        raise CauseFeasibilityError("legacy actual-LLM direct readback changed")
    v253 = legacy_llm["v253_development_walk_inventory_policy"]
    if v253["incremental_llm_edge_identified"] is not False \
            or v253["same_gross_cap"] is not False \
            or v253["same_cap_bot_control_present"] is not False \
            or v253["holdout"] != "UNOPENED":
        raise CauseFeasibilityError("V253 leverage confound classification changed")
    if legacy_llm["v15_jpy_portfolio_veto"]["stress_3x"]["net_pnl_jpy"] >= 0 \
            or legacy_llm["v18_aud_jpy_counterexample"]["normal_net_pnl_jpy"] >= 0 \
            or legacy_llm["v13_usd_major_inventory_policy"]["contained_additive_returns"] \
            ["ADVERSE_STRESS"] >= 0 \
            or legacy_llm["ai_test_bot_deterministic_not_actual_llm"] \
            ["actual_llm_attribution_proven"] is not False \
            or legacy_llm["broker_cohort_64d_101_trades"] \
            ["actual_llm_or_codex_attribution_proven"] is not False \
            or legacy_llm["current_v25_v41_actual_llm_status"] \
            ["machine_verified_actual_llm_strategy_arm_present"] is not False:
        raise CauseFeasibilityError("legacy actual-LLM evidence attribution changed")
    if payload["legacy_accounting_read_only_diagnostic"]["correction_reveals_hidden_2x"] is not False:
        raise CauseFeasibilityError("legacy accounting diagnostic falsely infers 2x")
    for entity in payload["full_month_summaries"].values():
        for arm in ("EXECUTABLE_BASE", "ADVERSE_STRESS"):
            if entity[arm]["all_full_months_at_least_2x"] is not False:
                raise CauseFeasibilityError("audit unexpectedly passes historical monthly gate")
    return payload


def main() -> int:
    root = Path(__file__).resolve().parent
    payload = build(root)
    validated = validate(root)
    if payload["audit_sha256"] != validated["audit_sha256"]:
        raise CauseFeasibilityError("audit build/readback mismatch")
    print(json.dumps({
        "audit_path": AUDIT_PATH,
        "audit_file_sha256": sha256_file(root / AUDIT_PATH),
        "audit_sha256": payload["audit_sha256"],
        "v42_current_execution_authorized": False,
        "selected_family": payload["v42_go_no_go"]["selected_family_id"],
        "holdout": payload["holdout_state"],
        "authority": AUTHORITY,
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
