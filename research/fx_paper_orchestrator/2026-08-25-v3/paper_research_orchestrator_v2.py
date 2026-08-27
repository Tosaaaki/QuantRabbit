#!/usr/bin/env python3
"""Fail-closed, restart-safe coordinator for local paper-only FX research.

The coordinator has no broker, account, credential, order, live, deploy or
external-configuration capability.  It binds preregistration, source BID/ASK
data and code before a single official subprocess is started, and separates
mechanical system acceptance from the much stricter strategy-profit gate.
"""

from __future__ import annotations

import argparse
import ast
import calendar
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


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
ARMS = ["RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS"]
PERIODS = ["WALK_FORWARD", "MONTH_2026_05", "MONTH_2026_06"]
PERIOD_BOUNDS = {
    "WALK_FORWARD": ("2026-05-01", "2026-07-01"),
    "MONTH_2026_05": ("2026-05-01", "2026-06-01"),
    "MONTH_2026_06": ("2026-06-01", "2026-07-01"),
}
V26_RECOVERY_WORK_ORDER = "V26_PRE_RESULT_RECOVERY_WORK_ORDER.json"
V26_RECOVERY_WORK_ORDER_SHA256 = "9f78f63ec2798bc38f701046ce0d21e1caa9fb65ff9391eca290b77efbec7ad1"
V26_RECOVERY_AUTHORIZATION = "V26_RECOVERY_AUTHORIZATION.json"
V26_RECOVERY_AUTHORIZATION_SHA256 = "c34e684ed2203d84025f9f560608c16baf513744e1fb37846c03264ca76d0256"
V26_RECOVERY_LAUNCHER = "run_causal_min_spread_representative_v26_recovery_once.py"
V26_RECOVERY_LAUNCHER_SHA256 = "8a137bd7f48facfd958a4d8e9c1977b84589d6016aa4ebcf4f413654a955430a"
V26_RECOVERY_FAILURE = "V26_AUTHORIZED_RECOVERY_FAILURE.json"
V26_RECOVERY_FAILURE_SHA256 = "75cceae96df7be5a51955a0966f587d378a0328ddf4f9c4f4947c2b3ed154a2b"
TERMINAL_NO_RERUN_STATUSES = {
    "FAILED_OFFICIAL_EXECUTION_NO_RERUN",
    "FAILED_AUTHORIZED_RECOVERY_NO_RERUN",
}
RAW_EDGE_REFINEMENT_POLICY_PATH = "RAW_EDGE_REFINEMENT_BUDGET_POLICY_V31.json"
RAW_EDGE_REFINEMENT_BUDGET = 3
RAW_EDGE_REFINEMENT_REASONS = {
    "EXECUTION_SUBSET_RAW_EDGE_ABSENT",
    "BASKET_HOLD_RAW_EDGE_ABSENT",
    "BASKET_CONSENSUS_RELEASE_RAW_EDGE_ABSENT",
    "CONSENSUS_RELEASE_SCOPE_RAW_EDGE_ABSENT",
    "CONSENSUS_RELEASE_PERSISTENCE_RAW_EDGE_ABSENT",
}
SIGNAL_FAMILY_PIVOT_REASON = "REPEATED_RAW_EDGE_ABSENT_SIGNAL_FAMILY_PIVOT"
SIGNAL_FAMILY_PIVOT_VARIABLE = (
    "one_preregistered_causal_fx_specific_signal_family_replacement_"
    "preserving_costs_leverage_periods_and_holdout"
)


class ContractError(RuntimeError):
    """A fail-closed contract violation."""


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


def sanitized_subprocess_excerpt(value: str, limit: int = 4000) -> str:
    """Keep bounded failure evidence without persisting local paths or secrets."""
    cleaned = "".join(character for character in value if character in "\n\t" or ord(character) >= 32)
    cleaned = re.sub(r"/Users/[^\s:'\"]+", "<local-path>", cleaned)
    cleaned = re.sub(
        r"(?i)\b(token|secret|password|credential)\s*[=:]\s*[^\s]+",
        r"\1=<redacted>",
        cleaned,
    )
    return cleaned[-limit:]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_journal(path: Path, event: str, **details: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "at": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **details,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def within(root: Path, relative: str | Path) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ContractError(f"path escapes owned research root: {relative}")
    return path


def require_keys(mapping: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        raise ContractError(f"{label} must be an object")
    missing = sorted(keys - set(mapping))
    if missing:
        raise ContractError(f"{label} missing required fields: {missing}")
    return mapping


def load_registry(root: Path, registry_path: Path) -> dict[str, Any]:
    path = within(root, registry_path)
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 2:
        raise ContractError("registry schema_version must be 2")
    if registry.get("authority") != AUTHORITY:
        raise ContractError("zero-authority paper-only contract mismatch")
    coordinator = require_keys(registry.get("coordinator"), {
        "path", "sha256", "failure_text_policy",
    }, "coordinator")
    coordinator_path = within(root, coordinator["path"])
    if not coordinator_path.is_file() or sha256_file(coordinator_path) != coordinator["sha256"]:
        raise ContractError("frozen coordinator hash mismatch")
    if coordinator["failure_text_policy"] != "SANITIZED_BOUNDED_EXCERPT_AND_FULL_SHA256":
        raise ContractError("coordinator failure evidence policy changed")
    legacy = require_keys(registry.get("legacy_evidence"), {
        "registry", "registry_sha256", "sealed_cycles", "policy",
    }, "legacy_evidence")
    legacy_path = within(root, legacy["registry"])
    if sha256_file(legacy_path) != legacy["registry_sha256"]:
        raise ContractError("legacy V1 registry changed; V4-V24 migration is not immutable")
    if legacy["policy"] != "READ_ONLY_MIGRATION_EVIDENCE_NO_RESULT_REWRITE":
        raise ContractError("legacy evidence rewrite policy is not fail-closed")
    if legacy["sealed_cycles"] != [f"V{number}" for number in range(4, 25)]:
        raise ContractError("legacy sealed cycle set must be exactly V4-V24")
    gate = require_keys(registry.get("profit_gate"), {
        "initial_equity_jpy", "full_comparable_month_required", "normal_min_multiple",
        "adverse_min_multiple", "stretch_multiple", "unopened_holdout_reproduction_required",
        "strategy_adoption_is_separate_gate", "forbidden",
    }, "profit_gate")
    if (gate["initial_equity_jpy"], gate["normal_min_multiple"], gate["adverse_min_multiple"],
            gate["stretch_multiple"]) != (200000, 2.0, 2.0, 3.0):
        raise ContractError("profit thresholds changed")
    if not gate["strategy_adoption_is_separate_gate"]:
        raise ContractError("system and strategy gates must remain separate")
    policy_ref = require_keys(registry.get("next_work_order_policy"), {
        "path", "sha256", "classification",
    }, "next_work_order_policy")
    if policy_ref["path"] != RAW_EDGE_REFINEMENT_POLICY_PATH \
            or policy_ref["classification"] != "NON_STRATEGY_ORCHESTRATOR_POLICY":
        raise ContractError("next-work-order policy identity changed")
    policy_path = within(root, policy_ref["path"])
    if not policy_path.is_file() or sha256_file(policy_path) != policy_ref["sha256"]:
        raise ContractError("next-work-order policy hash mismatch")
    validate_next_work_order_policy(root, json.loads(policy_path.read_text(encoding="utf-8")))
    cycles = registry.get("cycles")
    if not isinstance(cycles, list) or not cycles:
        raise ContractError("registry has no cycles")
    ids = [cycle.get("cycle_id") for cycle in cycles]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ContractError("cycle ids must be present and unique")
    for cycle in cycles:
        validate_cycle_contract(root, cycle)
    return registry


def validate_next_work_order_policy(root: Path, policy: dict[str, Any]) -> None:
    require_keys(policy, {
        "schema_version", "classification", "effective_from_parent_cycle",
        "max_consecutive_raw_edge_absent_refinements", "counted_reason_codes",
        "pivot_reason_code", "pivot_single_changed_variable", "grandfathered_work_order",
        "historical_derivation", "authority", "holdout",
    }, "next_work_order_policy")
    if policy["schema_version"] != 1 \
            or policy["classification"] != "NON_STRATEGY_ORCHESTRATOR_POLICY" \
            or policy["effective_from_parent_cycle"] != "V31" \
            or policy["max_consecutive_raw_edge_absent_refinements"] != RAW_EDGE_REFINEMENT_BUDGET:
        raise ContractError("raw-edge refinement budget policy changed")
    if set(policy["counted_reason_codes"]) != RAW_EDGE_REFINEMENT_REASONS:
        raise ContractError("raw-edge refinement reason set changed")
    if policy["pivot_reason_code"] != SIGNAL_FAMILY_PIVOT_REASON \
            or policy["pivot_single_changed_variable"] != SIGNAL_FAMILY_PIVOT_VARIABLE:
        raise ContractError("signal-family pivot route changed")
    if policy["authority"] != AUTHORITY \
            or policy["holdout"] != {"state": "UNOPENED", "may_execute": False}:
        raise ContractError("next-work-order policy authority or holdout changed")
    grandfathered = require_keys(policy["grandfathered_work_order"], {
        "path", "sha256", "parent_cycle", "proposed_cycle", "immutable",
    }, "grandfathered_work_order")
    grandfathered_path = within(root, grandfathered["path"])
    if grandfathered["parent_cycle"] != "V30" or grandfathered["proposed_cycle"] != "V31" \
            or grandfathered["immutable"] is not True \
            or not grandfathered_path.is_file() \
            or sha256_file(grandfathered_path) != grandfathered["sha256"]:
        raise ContractError("sealed V31 work order changed")
    history = policy["historical_derivation"]
    if [item.get("cycle_id") for item in history] != ["V27", "V28", "V29", "V30"]:
        raise ContractError("raw-edge refinement history must be V27-V30")
    state = read_state(within(root, "evidence/orchestrator_state_v2/state.json"))
    journal_path = within(root, "evidence/orchestrator_state_v2/failure_and_event_journal.jsonl")
    journal = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    for item in history:
        require_keys(item, {
            "cycle_id", "result", "result_sha256", "reason_code",
            "official_seal_sha256", "state_status", "journal_event",
        }, "historical_derivation item")
        result_path = within(root, item["result"])
        if not result_path.is_file() or sha256_file(result_path) != item["result_sha256"]:
            raise ContractError(f"historical result changed: {item['cycle_id']}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("automatic_rejection", {}).get("reason_code") != item["reason_code"] \
                or item["reason_code"] not in RAW_EDGE_REFINEMENT_REASONS:
            raise ContractError(f"historical RAW-edge reason changed: {item['cycle_id']}")
        state_record = state.get("cycles", {}).get(item["cycle_id"], {})
        if state_record.get("status") != item["state_status"] \
                or state_record.get("official_result_file_sha256") != item["result_sha256"] \
                or state_record.get("official_seal_sha256") != item["official_seal_sha256"]:
            raise ContractError(f"historical sealed state changed: {item['cycle_id']}")
        events = [event for event in journal
                  if event.get("event") == item["journal_event"]
                  and event.get("cycle_id") == item["cycle_id"]
                  and event.get("result_file_sha256") == item["result_sha256"]]
        if len(events) != 1:
            raise ContractError(f"historical seal journal changed: {item['cycle_id']}")


def validate_cycle_contract(root: Path, cycle: dict[str, Any]) -> None:
    require_keys(cycle, {
        "cycle_id", "registered_before_official_execution", "depends_on_legacy_cycle", "status",
        "preregistration", "preregistration_sha256", "script", "script_sha256", "test",
        "test_sha256", "dependency_files", "hypothesis_contract", "source_contract",
        "signal_contract", "inventory_contract", "evaluation_contract", "proposal_provenance",
        "execution",
    }, "cycle")
    if cycle["registered_before_official_execution"] is not True:
        raise ContractError("cycle was not registered before official execution")
    if cycle["status"] != "REGISTERED_PENDING_OFFICIAL_EXECUTION":
        raise ContractError("registry cycle status is not the immutable preregistered state")
    for name, hash_name in (("preregistration", "preregistration_sha256"),
                            ("script", "script_sha256"), ("test", "test_sha256")):
        path = within(root, cycle[name])
        if not path.is_file() or sha256_file(path) != cycle[hash_name]:
            raise ContractError(f"frozen {name} hash mismatch for {cycle['cycle_id']}")
    for dependency in cycle["dependency_files"]:
        require_keys(dependency, {"path", "sha256"}, "dependency")
        path = within(root, dependency["path"])
        if not path.is_file() or sha256_file(path) != dependency["sha256"]:
            raise ContractError(f"dependency hash mismatch: {dependency['path']}")

    hypothesis = require_keys(cycle["hypothesis_contract"], {
        "independent", "frozen_before_replay", "baseline", "family", "family_hypothesis_number",
        "single_changed_variable", "changed_variable_count",
    }, "hypothesis_contract")
    if hypothesis["independent"] is not True or hypothesis["frozen_before_replay"] is not True:
        raise ContractError("hypothesis is not independently frozen")
    if hypothesis["changed_variable_count"] != 1 or not hypothesis["single_changed_variable"]:
        raise ContractError("exactly one changed variable is required")

    source = require_keys(cycle["source_contract"], {
        "root", "bar_granularity", "price_component", "completed_only",
        "strictly_increasing_timestamp", "files", "manifest_sha256",
    }, "source_contract")
    if source["price_component"] != "BID_ASK" or source["completed_only"] is not True:
        raise ContractError("actual completed BID/ASK source is required")
    if source["strictly_increasing_timestamp"] is not True:
        raise ContractError("completed-data chronology must be enforced")
    if hashlib.sha256(canonical_bytes(source["files"])).hexdigest() != source["manifest_sha256"]:
        raise ContractError("source manifest seal mismatch")

    signal = require_keys(cycle["signal_contract"], {
        "arms", "same_signal_id_set", "same_entry_direction_exit", "raw_cost_gate",
        "base_cost_model", "adverse_cost_model",
    }, "signal_contract")
    if signal["arms"] != ARMS or signal["same_signal_id_set"] is not True:
        raise ContractError("all three cost arms must share one signal_id set")
    if signal["same_entry_direction_exit"] is not True or signal["raw_cost_gate"] is not False:
        raise ContractError("RAW signal must be identical and independent of costs")
    tree = ast.parse(within(root, cycle["script"]).read_text(encoding="utf-8"))
    raw_source = signal.get("raw_signal_source", "GENERATED_IN_CYCLE")
    if raw_source == "GENERATED_IN_CYCLE":
        detectors = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "detect_day_signals"]
        if len(detectors) != 1 or [arg.arg for arg in detectors[0].args.args] != ["pair_day_bars"]:
            raise ContractError("signal detector exposes cost or outcome inputs")
        if cycle["cycle_id"] != "V33":
            simulator_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                               and isinstance(node.func, ast.Name) and node.func.id == "simulate_portfolio"]
            if len(simulator_calls) != 1 or len(simulator_calls[0].args) < 3:
                raise ContractError("cost-arm simulation call is not structurally unique")
            if not isinstance(simulator_calls[0].args[1], ast.Name) \
                    or simulator_calls[0].args[1].id != "rows":
                raise ContractError("cost arms do not consume the same RAW_SIGNAL ledger")
        if cycle["cycle_id"] == "V32":
            prereg = json.loads(within(root, cycle["preregistration"]).read_text(encoding="utf-8"))
            predecessor = prereg.get("predecessor_disposition", {})
            selection = prereg.get("training_only_family_selection", {})
            signal_rule = prereg.get("signal_family_rule", {})
            execution_rule = prereg.get("frozen_execution_contract", {})
            if predecessor.get("status") != "FROZEN_REJECTED_EVIDENCE_NO_REWRITE_NO_RERUN" \
                    or predecessor.get("reason_code") != "CONSENSUS_RELEASE_PERSISTENCE_RAW_EDGE_ABSENT" \
                    or predecessor.get("work_order_sha256") \
                    != "82ed0e702c7691ce424ffeb75283c4a711356565f36ea59c3a454696f47b4d26":
                raise ContractError("V32 did not preserve V31 and its signal-family pivot work order")
            if selection.get("candidate_signal_families_preregistered") != 1 \
                    or selection.get("candidate_signal_families_compared_by_outcome") != 0 \
                    or selection.get("post_entry_return_outcome_consulted") is not False \
                    or selection.get("cost_consulted") is not False \
                    or selection.get("evaluation_month_used_for_selection") is not False \
                    or selection.get("walk_forward_used_for_selection") is not False \
                    or selection.get("holdout_used") is not False:
                raise ContractError("V32 signal family was not one training-only outcome-free candidate")
            if signal_rule.get("name") != "PAIR_SPECIFIC_ASIAN_TAIL_DISPLACEMENT_HANDOFF_FADE" \
                    or signal_rule.get("direction_formula") != "native_pair_direction = -sign(d[pair,day])" \
                    or signal_rule.get("maximum_signals_per_pair_utc_day") != 1 \
                    or signal_rule.get("cost_or_post_entry_outcome_inputs") is not False:
                raise ContractError("V32 signal-family rule differs from preregistration")
            if execution_rule.get("baseline") != "V31 inventory and exit state machine" \
                    or execution_rule.get("changed_from_v31") is not False \
                    or execution_rule.get("required_consecutive_confirmations") != 2 \
                    or execution_rule.get("target_hold_seconds") != 172800 \
                    or execution_rule.get("hard_max_age_seconds") \
                    != cycle["inventory_contract"]["finite_max_age_seconds"] \
                    or execution_rule.get("cost_or_outcome_inputs") is not False:
                raise ContractError("V32 changed the frozen V31 execution state machine")
        if cycle["cycle_id"] == "V33":
            prereg = json.loads(within(root, cycle["preregistration"]).read_text(encoding="utf-8"))
            failure = prereg.get("failed_predecessor", {})
            runtime = prereg.get("runtime_compatibility_provenance", {})
            strategy = prereg.get("frozen_v32_signal_family", {})
            if failure.get("cycle_id") != "V32" \
                    or failure.get("status") != "FAILED_OFFICIAL_EXECUTION_NO_RESULT_NO_RERUN" \
                    or failure.get("metrics_available") is not False \
                    or failure.get("rerun_permitted") is not False:
                raise ContractError("V33 did not preserve V32 as a terminal pre-result failure")
            if runtime.get("classification") != "NON_STRATEGY_RUNTIME_COMPATIBILITY" \
                    or runtime.get("changed_strategy_variables") != 0 \
                    or runtime.get("integer_epoch_nanoseconds_changed") is not False \
                    or runtime.get("signal_ids_changed") is not False \
                    or runtime.get("directions_changed") is not False \
                    or runtime.get("v32_rerun_permitted") is not False:
                raise ContractError("V33 timestamp recovery changed the unobserved V32 strategy")
            if strategy.get("source_script_sha256") \
                    != "349447aa53a1dc8ff837e29960326b9c24b3276083f78a273dfceaf3c685eaca" \
                    or strategy.get("candidate_signal_families_compared_by_outcome") != 0 \
                    or strategy.get("cost_or_post_entry_outcome_inputs") is not False:
                raise ContractError("V33 did not freeze the V32 signal family")
            canonicalizers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                              and node.name == "canonical_utc_nine_digits"]
            if len(canonicalizers) != 1 or [arg.arg for arg in canonicalizers[0].args.args] != ["value"]:
                raise ContractError("V33 timestamp canonicalizer contract mismatch")
    elif raw_source == "SEALED_PARENT_V25_LEDGER":
        require_keys(signal, {
            "parent_cycle_id", "parent_ledger", "parent_ledger_sha256", "parent_signal_id_set_sha256",
            "same_decision_timestamps", "same_execution_mask_all_arms",
        }, "parent signal_contract")
        if signal["parent_cycle_id"] != "V25" or signal["same_decision_timestamps"] is not True:
            raise ContractError("V26 parent RAW identity contract mismatch")
        if signal["same_execution_mask_all_arms"] is not True:
            raise ContractError("V26 cost arms must share one execution mask")
        parent_ledger = within(root, signal["parent_ledger"])
        if not parent_ledger.is_file() or sha256_file(parent_ledger) != signal["parent_ledger_sha256"]:
            raise ContractError("V26 sealed parent ledger dependency mismatch")
        if cycle["cycle_id"] in {"V26", "V27"}:
            selectors = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name == "apply_rule"]
            scorers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "causal_score"]
            if len(selectors) != 1 or [arg.arg for arg in selectors[0].args.args] != ["parent_rows", "corpus"]:
                raise ContractError("V26/V27 deterministic execution selector contract mismatch")
            if len(scorers) != 1 or [arg.arg for arg in scorers[0].args.args] != ["row", "bars", "time_index"]:
                raise ContractError("V26/V27 causal cost score contract mismatch")
        elif cycle["cycle_id"] == "V28":
            builders = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "build_execution_ledger"]
            if len(builders) != 1 or [arg.arg for arg in builders[0].args.args] != ["parent_rows", "corpus"]:
                raise ContractError("V28 deterministic basket-hold builder contract mismatch")
            prereg = json.loads(within(root, cycle["preregistration"]).read_text(encoding="utf-8"))
            predecessor = prereg.get("predecessor_disposition", {})
            rule = prereg.get("execution_rule", {})
            if predecessor.get("status") != "FROZEN_REJECTED_EVIDENCE_NO_REWRITE_NO_RERUN" \
                    or predecessor.get("reason_code") != "EXECUTION_SUBSET_RAW_EDGE_ABSENT":
                raise ContractError("V28 did not preserve V27 as frozen rejected evidence")
            if prereg.get("training_only_rule_selection", {}).get("candidate_rules_compared") != 1 \
                    or prereg["training_only_rule_selection"].get("return_outcome_consulted") is not False \
                    or prereg["training_only_rule_selection"].get("cost_consulted") is not False:
                raise ContractError("V28 rule was not selected as one training-only outcome-free candidate")
            if rule.get("same_direction_add_units") != 0 \
                    or rule.get("same_direction_expiry_extension_seconds") != 0 \
                    or rule.get("hard_max_age_seconds") != cycle["inventory_contract"]["finite_max_age_seconds"]:
                raise ContractError("V28 basket-hold no-add/max-age rule differs from preregistration")
        elif cycle["cycle_id"] == "V29":
            builders = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "build_execution_ledger"]
            voters = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "consensus_vote"]
            if len(builders) != 1 or [arg.arg for arg in builders[0].args.args] != ["parent_rows", "corpus"]:
                raise ContractError("V29 deterministic consensus-release builder contract mismatch")
            if len(voters) != 1 or [arg.arg for arg in voters[0].args.args] != ["signals", "held_pair"]:
                raise ContractError("V29 deterministic peer consensus contract mismatch")
            prereg = json.loads(within(root, cycle["preregistration"]).read_text(encoding="utf-8"))
            predecessor = prereg.get("predecessor_disposition", {})
            selection = prereg.get("training_only_rule_selection", {})
            rule = prereg.get("execution_rule", {})
            if predecessor.get("status") != "FROZEN_REJECTED_EVIDENCE_NO_REWRITE_NO_RERUN" \
                    or predecessor.get("reason_code") != "BASKET_HOLD_RAW_EDGE_ABSENT":
                raise ContractError("V29 did not preserve V28 as frozen rejected evidence")
            if selection.get("candidate_rules_compared") != 1 \
                    or selection.get("return_outcome_consulted") is not False \
                    or selection.get("cost_consulted") is not False \
                    or selection.get("walk_forward_used_for_rule_selection") is not False:
                raise ContractError("V29 rule was not selected as one training-only outcome-free candidate")
            if rule.get("minimum_peer_signals") != 2 \
                    or rule.get("unanimity_required") is not True \
                    or rule.get("same_direction_add_units") != 0 \
                    or rule.get("same_direction_expiry_extension_seconds") != 0 \
                    or rule.get("hard_max_age_seconds") != cycle["inventory_contract"]["finite_max_age_seconds"]:
                raise ContractError("V29 consensus release rule differs from preregistration")
        elif cycle["cycle_id"] == "V30":
            builders = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "build_execution_ledger"]
            scopes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "scoped_peer_signals"]
            if len(builders) != 1 or [arg.arg for arg in builders[0].args.args] != ["parent_rows", "corpus"]:
                raise ContractError("V30 deterministic scoped-release builder contract mismatch")
            if len(scopes) != 1 or [arg.arg for arg in scopes[0].args.args] != [
                    "simultaneous_signals", "target_pair", "position_snapshot"]:
                raise ContractError("V30 deterministic peer-scope contract mismatch")
            prereg = json.loads(within(root, cycle["preregistration"]).read_text(encoding="utf-8"))
            predecessor = prereg.get("predecessor_disposition", {})
            selection = prereg.get("training_only_scope_selection", {})
            rule = prereg.get("peer_scope_rule", {})
            if predecessor.get("status") != "FROZEN_REJECTED_EVIDENCE_NO_REWRITE_NO_RERUN" \
                    or predecessor.get("reason_code") != "BASKET_CONSENSUS_RELEASE_RAW_EDGE_ABSENT":
                raise ContractError("V30 did not preserve V29 as frozen rejected evidence")
            if selection.get("scope_count_preregistered") != 1 \
                    or selection.get("candidate_scopes_compared_by_outcome") != 0 \
                    or selection.get("price_consulted") is not False \
                    or selection.get("return_outcome_consulted") is not False \
                    or selection.get("cost_consulted") is not False \
                    or selection.get("evaluation_month_used_for_scope_selection") is not False:
                raise ContractError("V30 scope was not selected as one training-only outcome-free candidate")
            if rule.get("name") != "ACTIVE_SAME_SIGNED_USD_INVENTORY_SUBGRAPH" \
                    or rule.get("only_changed_field_from_v29") != "peer membership scope" \
                    or rule.get("minimum_peer_signals") != 2 \
                    or rule.get("unanimity_required") is not True \
                    or rule.get("same_timestamp_required") is not True \
                    or rule.get("self_pair_excluded") is not True \
                    or rule.get("hard_max_age_seconds") != cycle["inventory_contract"]["finite_max_age_seconds"] \
                    or rule.get("cost_or_outcome_inputs") is not False:
                raise ContractError("V30 peer scope differs from preregistration")
        elif cycle["cycle_id"] == "V31":
            builders = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "build_execution_ledger"]
            planners = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "build_period_plans"]
            if len(builders) != 1 or [arg.arg for arg in builders[0].args.args] != [
                    "parent_rows", "corpus"]:
                raise ContractError("V31 deterministic persistence builder contract mismatch")
            if len(planners) != 1 or [arg.arg for arg in planners[0].args.args] != [
                    "corpus", "parent_rows", "start", "end"]:
                raise ContractError("V31 deterministic persistence planner contract mismatch")
            prereg = json.loads(within(root, cycle["preregistration"]).read_text(encoding="utf-8"))
            predecessor = prereg.get("predecessor_disposition", {})
            selection = prereg.get("training_only_persistence_selection", {})
            rule = prereg.get("persistence_confirmation_rule", {})
            system_policy = prereg.get("non_strategy_orchestrator_contract", {})
            if predecessor.get("status") != "FROZEN_REJECTED_EVIDENCE_NO_REWRITE_NO_RERUN" \
                    or predecessor.get("reason_code") != "CONSENSUS_RELEASE_SCOPE_RAW_EDGE_ABSENT":
                raise ContractError("V31 did not preserve V30 as frozen rejected evidence")
            if selection.get("candidate_rules_preregistered") != 1 \
                    or selection.get("candidate_confirmation_counts_compared_by_outcome") != 0 \
                    or selection.get("required_consecutive_confirmations") != 2 \
                    or selection.get("price_consulted") is not False \
                    or selection.get("return_outcome_consulted") is not False \
                    or selection.get("cost_consulted") is not False \
                    or selection.get("evaluation_month_used_for_selection") is not False:
                raise ContractError("V31 persistence rule was not one training-only outcome-free candidate")
            if rule.get("name") != "TWO_CONSECUTIVE_COMPLETED_DECISION_EVENTS_SAME_USD_CONSENSUS" \
                    or rule.get("only_changed_field_from_v30") != "required consecutive confirmation events" \
                    or rule.get("required_consecutive_confirmations") != 2 \
                    or rule.get("peer_scope") != "ACTIVE_SAME_SIGNED_USD_INVENTORY_SUBGRAPH" \
                    or rule.get("minimum_peer_signals") != 2 \
                    or rule.get("unanimity_required") is not True \
                    or rule.get("same_timestamp_required") is not True \
                    or rule.get("self_pair_excluded") is not True \
                    or rule.get("direction_formula_changed_from_v30") is not False \
                    or not str(rule.get("finite_max_age_precedence", "")) \
                    or rule.get("hard_max_age_seconds") != cycle["inventory_contract"]["finite_max_age_seconds"] \
                    or rule.get("cost_or_outcome_inputs") is not False:
                raise ContractError("V31 persistence rule differs from preregistration")
            if system_policy.get("classification") != "NON_STRATEGY_ORCHESTRATOR_POLICY" \
                    or system_policy.get("changed_strategy_variables") != 0 \
                    or system_policy.get("changes_v31_signal_action_or_result") is not False \
                    or system_policy.get("grandfathered_v31_work_order_sha256") \
                    != "873d6d92b9f18b66b9ded21f339d643bfd16b0f6bda6759905f95f36ab6b8763":
                raise ContractError("V31 strategy and successor-routing policy contracts were mixed")
        else:
            raise ContractError(f"unsupported sealed-parent execution cycle: {cycle['cycle_id']}")
        if cycle["cycle_id"] == "V27":
            prereg = json.loads(within(root, cycle["preregistration"]).read_text(encoding="utf-8"))
            runtime = prereg.get("runtime_compatibility_provenance", {})
            if runtime.get("classification") != "NON_STRATEGY_RUNTIME_COMPATIBILITY" \
                    or runtime.get("changed_strategy_variables") != 0 \
                    or runtime.get("v26_rerun_permitted") is not False:
                raise ContractError("V27 runtime compatibility is mixed with strategy authority")
            if prereg.get("hypothesis_contract", {}).get("same_unobserved_strategy_as_v26") is not True:
                raise ContractError("V27 did not preserve the unobserved V26 strategy")
            failure = within(root, V26_RECOVERY_FAILURE)
            if not failure.is_file() or sha256_file(failure) != V26_RECOVERY_FAILURE_SHA256:
                raise ContractError("V27 predecessor failure evidence changed")
    else:
        raise ContractError(f"unsupported raw signal source: {raw_source}")

    inventory = require_keys(cycle["inventory_contract"], {
        "currencies", "one_position_per_pair", "one_basket_per_utc_day", "add_to_position",
        "martingale", "overlap", "fixed_pair_sleeve", "gross_leverage_cap",
        "currency_abs_exposure_cap", "finite_max_age_seconds", "terminal_liquidation_required",
        "terminal_mtm_hidden",
    }, "inventory_contract")
    if any(inventory[key] is not True for key in (
        "one_position_per_pair", "one_basket_per_utc_day", "terminal_liquidation_required")):
        raise ContractError("inventory and terminal liquidation requirements are incomplete")
    if any(inventory[key] is not False for key in ("add_to_position", "martingale", "overlap",
                                                   "terminal_mtm_hidden")):
        raise ContractError("inventory escalation or hidden MTM is forbidden")
    if not isinstance(inventory["finite_max_age_seconds"], int) or inventory["finite_max_age_seconds"] <= 0:
        raise ContractError("finite positive max-age is required")
    if not 0 < inventory["gross_leverage_cap"] <= 1 or not 0 < inventory["currency_abs_exposure_cap"] <= 1:
        raise ContractError("paper inventory exposure caps must be in (0, 1]")

    evaluation = require_keys(cycle["evaluation_contract"], {
        "walk_forward", "full_comparable_months", "holdout", "development_evidence_label",
    }, "evaluation_contract")
    holdout = require_keys(evaluation["holdout"], {"label", "state", "may_execute"}, "holdout")
    if holdout["state"] != "UNOPENED" or holdout["may_execute"] is not False:
        raise ContractError("holdout must remain labelled and unopened")
    if evaluation["full_comparable_months"] != PERIODS[1:]:
        raise ContractError("full comparable monthly periods changed")

    provenance = require_keys(cycle["proposal_provenance"], {
        "proposal_kind", "policy", "policy_provenance", "model_identity",
        "unverified_identity_may_not_be_inferred",
    }, "proposal_provenance")
    if provenance["unverified_identity_may_not_be_inferred"] is not True:
        raise ContractError("unverified LLM identity must not be inferred")

    execution = require_keys(cycle["execution"], {
        "argv", "pythonpath", "timeout_seconds", "official_run_limit", "result", "ledger",
    }, "execution")
    if execution["official_run_limit"] != 1 or execution["timeout_seconds"] <= 0:
        raise ContractError("official execution must be bounded to one attempt")
    if execution["argv"][0] != cycle["script"] or "--input-root" not in execution["argv"]:
        raise ContractError("execution argv is not bound to the frozen script/source")
    for item in execution["pythonpath"]:
        within(root, item)
    within(root, execution["result"])
    within(root, execution["ledger"])


def validate_source(cycle: dict[str, Any]) -> dict[str, Any]:
    source = cycle["source_contract"]
    source_root = Path(source["root"])
    audit: dict[str, Any] = {}
    for pair, expected_hash in sorted(source["files"].items()):
        matches = sorted((source_root / pair).glob("*_M5_BA_*.jsonl.gz"))
        if len(matches) != 1:
            raise ContractError(f"expected exactly one BID/ASK source for {pair}")
        path = matches[0]
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ContractError(f"source BID/ASK hash mismatch for {pair}")
        previous = ""
        rows = 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                raw = json.loads(line)
                if raw.get("complete") is not True or raw.get("price") != "BA":
                    raise ContractError(f"non-completed/non-BIDASK source row {pair}:{line_no}")
                stamp = str(raw.get("time", ""))
                if stamp <= previous:
                    raise ContractError(f"non-increasing source chronology {pair}:{line_no}")
                previous = stamp
                if set(raw.get("bid", {})) < set("ohlc") or set(raw.get("ask", {})) < set("ohlc"):
                    raise ContractError(f"missing BID/ASK OHLC source fields {pair}:{line_no}")
                if any(float(raw["ask"][key]) < float(raw["bid"][key]) for key in "ohlc"):
                    raise ContractError(f"crossed BID/ASK source row {pair}:{line_no}")
                rows += 1
        if rows < 100:
            raise ContractError(f"source unexpectedly short for {pair}")
        audit[pair] = {"sha256": actual_hash, "rows": rows, "last_timestamp": previous}
    return audit


@dataclass(frozen=True, order=True)
class EpochNanoseconds:
    value: int

    def __sub__(self, other: "EpochNanoseconds") -> "NanosecondDelta":
        if not isinstance(other, EpochNanoseconds):
            return NotImplemented
        return NanosecondDelta(self.value - other.value)


@dataclass(frozen=True)
class NanosecondDelta:
    value: int

    def total_seconds(self) -> float:
        return self.value / 1_000_000_000


_UTC_TIMESTAMP = re.compile(
    r"^(?P<head>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?Z$"
)


def parse_time(stamp: str) -> EpochNanoseconds:
    """Preserve canonical UTC ordering and elapsed time to integer nanoseconds."""
    match = _UTC_TIMESTAMP.fullmatch(stamp)
    if match is None:
        raise ContractError(f"timestamp is not canonical explicit UTC: {stamp}")
    try:
        seconds = datetime.strptime(match.group("head"), "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ContractError(f"invalid timestamp: {stamp}") from error
    fraction = match.group("fraction") or ""
    return EpochNanoseconds(
        calendar.timegm(seconds.utctimetuple()) * 1_000_000_000
        + int(fraction.ljust(9, "0") or "0")
    )


def validate_result(root: Path, cycle: dict[str, Any]) -> dict[str, Any]:
    execution = cycle["execution"]
    result_path = within(root, execution["result"])
    ledger_path = within(root, execution["ledger"])
    if not result_path.is_file() or not ledger_path.is_file():
        raise ContractError("official result or proposal ledger is missing")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("result_sha256") != embedded_hash(payload, "result_sha256"):
        raise ContractError("frozen result embedded seal mismatch")
    if payload.get("live_authority") is not False or payload.get("external_orders") != 0:
        raise ContractError("result violates zero live/order authority")
    if payload.get("terminal_inventory_mtm_hidden") is not False:
        raise ContractError("result does not explicitly reject hidden terminal MTM")
    if payload.get("cost_suppressed_raw_signals") != 0:
        raise ContractError("RAW signal stream was cost-gated")
    if payload.get("same_signal_stream_all_cost_arms") is not True:
        raise ContractError("cost arms did not share one signal stream")
    if payload.get("final_admitted") is not False:
        raise ContractError("development result cannot admit a final strategy")
    if payload.get("evidence_class") != "opened_development_not_future_holdout":
        raise ContractError("opened development evidence was mislabeled")
    portfolio = payload.get("portfolio", {})
    expected_inventory = cycle["inventory_contract"]
    if portfolio.get("gross_leverage_cap") != expected_inventory["gross_leverage_cap"]:
        raise ContractError("gross leverage cap changed after preregistration")
    if not math.isclose(float(portfolio.get("weight_per_pair", -1)),
                        float(expected_inventory["fixed_pair_sleeve"]), rel_tol=0, abs_tol=1e-15):
        raise ContractError("fixed pair sleeve changed")

    ledger_hash = sha256_file(ledger_path)
    if ledger_hash != payload.get("proposal_ledger_sha256"):
        raise ContractError("proposal ledger hash mismatch")
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
    ids: list[str] = []
    previous_key: tuple[str, str] | None = None
    seen_day_pair: set[tuple[str, str]] = set()
    max_age = cycle["inventory_contract"]["finite_max_age_seconds"]
    for row in rows:
        required = {"signal_id", "pair", "utc_day", "decision_time", "fill_time", "exit_time", "direction"}
        if set(row) < required:
            raise ContractError("proposal ledger row lacks causal identity fields")
        decision, fill, exit_time = map(parse_time, (row["decision_time"], row["fill_time"], row["exit_time"]))
        if not decision < fill <= exit_time:
            raise ContractError("completed-data chronology or fill/exit order failed")
        if (exit_time - fill).total_seconds() > max_age:
            raise ContractError("finite max-age exceeded")
        day_pair = (row["utc_day"], row["pair"])
        if day_pair in seen_day_pair:
            raise ContractError("more than one position per pair/day")
        seen_day_pair.add(day_pair)
        key = (row["fill_time"], row["signal_id"])
        if previous_key is not None and key < previous_key:
            raise ContractError("proposal ledger is not deterministically ordered")
        previous_key = key
        ids.append(row["signal_id"])
    if len(ids) != len(set(ids)) or len(ids) != payload.get("raw_signals"):
        raise ContractError("signal_id set is not unique or does not match result")
    sleeve = float(cycle["inventory_contract"]["fixed_pair_sleeve"])
    currency_cap = float(cycle["inventory_contract"]["currency_abs_exposure_cap"])
    by_day: dict[str, list[dict[str, Any]]] = {}
    inventory_rows = [row for row in rows if row.get("execution_selected", True) is True]
    for row in inventory_rows:
        by_day.setdefault(row["utc_day"], []).append(row)
    selected_per_basket = cycle["inventory_contract"].get("selected_positions_per_basket")
    if selected_per_basket is not None:
        raw_days = {row["utc_day"] for row in rows}
        if selected_per_basket != 1 or set(by_day) != raw_days \
                or any(len(day_rows) != 1 for day_rows in by_day.values()):
            raise ContractError("deterministic representative rule did not select exactly one position per basket")
    for day, day_rows in by_day.items():
        if len(day_rows) * sleeve > cycle["inventory_contract"]["gross_leverage_cap"] + 1e-12:
            raise ContractError(f"gross inventory cap exceeded on {day}")
        exposures: dict[str, float] = {}
        for row in day_rows:
            base, quote = row["pair"].split("_")
            signed = sleeve * int(row["direction"])
            exposures[base] = exposures.get(base, 0.0) + signed
            exposures[quote] = exposures.get(quote, 0.0) - signed
        if any(abs(value) > currency_cap + 1e-12 for value in exposures.values()):
            raise ContractError(f"currency exposure cap exceeded on {day}")

    source_audit = {item["pair"]: item["source_sha256"] for item in payload.get("source_audit", [])}
    if source_audit != cycle["source_contract"]["files"]:
        raise ContractError("result source BID/ASK manifest differs from preregistration")
    raw_source = cycle.get("signal_contract", {}).get("raw_signal_source", "GENERATED_IN_CYCLE")
    if raw_source == "SEALED_PARENT_V25_LEDGER":
        parent_rows = [json.loads(line) for line in within(
            root, cycle["signal_contract"]["parent_ledger"]
        ).read_text(encoding="utf-8").splitlines() if line]
        identity_fields = ("signal_id", "pair", "utc_day", "direction", "decision_time", "fill_time", "exit_time")
        if [[row[field] for field in identity_fields] for row in rows] != [
                [row[field] for field in identity_fields] for row in parent_rows]:
            raise ContractError("V26 ledger changed V25 RAW identity, direction, or timestamps")
        if hashlib.sha256(canonical_bytes(sorted(ids))).hexdigest() \
                != cycle["signal_contract"]["parent_signal_id_set_sha256"]:
            raise ContractError("V26 ledger signal-id set differs from V25")
        if payload.get("parent_ledger_sha256") != cycle["signal_contract"]["parent_ledger_sha256"]:
            raise ContractError("V26 parent ledger hash mismatch in result")
        if payload.get("parent_signal_id_set_sha256") != cycle["signal_contract"]["parent_signal_id_set_sha256"]:
            raise ContractError("V26 parent signal-id hash mismatch in result")
        if payload.get("same_parent_signal_id_set") is not True \
                or payload.get("same_parent_decision_timestamps") is not True:
            raise ContractError("sealed-parent cycle changed the V25 RAW signal identity or decision timestamps")
        action_material = []
        for row in rows:
            actions = row.get("arm_actions")
            if not isinstance(actions, dict) or set(actions) != set(ARMS) \
                    or len(set(actions.values())) != 1:
                raise ContractError("sealed-parent ledger arm execution actions differ")
            action_material.append([row["signal_id"], row.get("execution_action")])
        if cycle["cycle_id"] in {"V26", "V27"}:
            if payload.get("same_execution_mask_all_cost_arms") is not True:
                raise ContractError("V26/V27 cost arms do not share one execution mask")
            mask = [[row["signal_id"], row.get("execution_selected") is True] for row in rows]
            if hashlib.sha256(canonical_bytes(mask)).hexdigest() != payload.get("execution_mask_sha256"):
                raise ContractError("V26/V27 execution mask hash mismatch")
        elif cycle["cycle_id"] in {"V28", "V29", "V30", "V31"}:
            expected_experiment = {
                "V28": "FX_CAUSAL_BASKET_HOLD_V28",
                "V29": "FX_CAUSAL_BASKET_CONSENSUS_RELEASE_V29",
                "V30": "FX_CAUSAL_CONSENSUS_RELEASE_SCOPE_V30",
                "V31": "FX_CAUSAL_CONSENSUS_RELEASE_PERSISTENCE_V31",
            }[cycle["cycle_id"]]
            if payload.get("cycle_id") != cycle["cycle_id"] \
                    or payload.get("experiment") != expected_experiment \
                    or payload.get("same_execution_state_transitions_all_cost_arms") is not True \
                    or payload.get("same_parent_directions") is not True:
                raise ContractError(f"{cycle['cycle_id']} result identity or execution-state parity mismatch")
            if any(row.get("execution_selected") is not True for row in rows):
                raise ContractError(f"{cycle['cycle_id']} removed a V25 RAW signal from the state ledger")
            if hashlib.sha256(canonical_bytes(action_material)).hexdigest() \
                    != payload.get("execution_action_sha256"):
                raise ContractError(f"{cycle['cycle_id']} execution action hash mismatch")
            rule = payload.get("execution_rule", {})
            if rule.get("cost_or_outcome_inputs") is not False \
                    or rule.get("hard_max_age_seconds") != max_age:
                raise ContractError(f"{cycle['cycle_id']} execution rule used a forbidden input or changed max-age")
            if cycle["cycle_id"] == "V29" and (
                    rule.get("minimum_peer_signals") != 2
                    or rule.get("unanimity_required") is not True
                    or rule.get("own_pair_signal_prevents_consensus_release") is not True):
                raise ContractError("V29 result consensus rule differs from preregistration")
            if cycle["cycle_id"] == "V30" and (
                    rule.get("peer_scope") != "ACTIVE_SAME_SIGNED_USD_INVENTORY_SUBGRAPH"
                    or rule.get("only_changed_field_from_v29") != "peer_membership_scope"
                    or rule.get("minimum_peer_signals") != 2
                    or rule.get("unanimity_required") is not True
                    or rule.get("same_timestamp_required") is not True
                    or rule.get("self_pair_excluded") is not True
                    or rule.get("direction_formula_changed_from_v29") is not False
                    or rule.get("own_pair_signal_prevents_consensus_release") is not True):
                raise ContractError("V30 result peer scope differs from preregistration")
            if cycle["cycle_id"] == "V31" and (
                    rule.get("only_changed_field_from_v30") != "required_consecutive_confirmation_events"
                    or rule.get("required_consecutive_confirmations") != 2
                    or rule.get("confirmation_unit") != "completed_global_V25_decision_event"
                    or rule.get("peer_scope") != "ACTIVE_SAME_SIGNED_USD_INVENTORY_SUBGRAPH"
                    or rule.get("minimum_peer_signals") != 2
                    or rule.get("unanimity_required") is not True
                    or rule.get("same_timestamp_required") is not True
                    or rule.get("self_pair_excluded") is not True
                    or rule.get("direction_formula_changed_from_v30") is not False
                    or rule.get("own_pair_signal_prevents_consensus_release") is not True
                    or rule.get("finite_max_age_precedence") is not True):
                raise ContractError("V31 result persistence rule differs from preregistration")
        if cycle["cycle_id"] == "V27":
            runtime = payload.get("runtime_compatibility_provenance", {})
            if payload.get("cycle_id") != "V27" \
                    or payload.get("experiment") != "FX_CAUSAL_MIN_SPREAD_REPRESENTATIVE_V27" \
                    or runtime.get("classification") != "NON_STRATEGY_RUNTIME_COMPATIBILITY" \
                    or runtime.get("changed_strategy_variables") != 0 \
                    or runtime.get("v26_rerun_permitted") is not False:
                raise ContractError("V27 result runtime provenance or cycle identity mismatch")

    if cycle["cycle_id"] in {"V32", "V33"}:
        expected_experiment = f"FX_ASIAN_DISPLACEMENT_HANDOFF_FADE_{cycle['cycle_id']}"
        if payload.get("cycle_id") != cycle["cycle_id"] \
                or payload.get("experiment") != expected_experiment \
                or payload.get("single_changed_variable") \
                != "fx_specific_asian_displacement_handoff_fade_signal_family" \
                or payload.get("same_execution_actions_all_cost_arms") is not True \
                or payload.get("same_execution_state_transitions_all_cost_arms") is not True:
            raise ContractError(f"{cycle['cycle_id']} result identity or cost-arm parity mismatch")
        if payload.get("holdout") != {
                "label": "FUTURE_FX_HOLDOUT_AFTER_2026_07_15",
                "state": "UNOPENED", "may_execute": False}:
            raise ContractError(f"{cycle['cycle_id']} holdout label or unopened state changed")
        indicator = payload.get("indicator", {})
        if indicator.get("training_abs_displacement_quantile") != 0.75 \
                or indicator.get("direction_formula") \
                != "-sign(log(mid_close_05:55 / mid_open_00:00))" \
                or indicator.get("cost_used_for_signal") is not False \
                or indicator.get("post_entry_outcome_used_for_signal") is not False \
                or indicator.get("evaluation_month_used_for_threshold") is not False:
            raise ContractError(f"{cycle['cycle_id']} result signal formula differs from preregistration")
        execution_rule = payload.get("execution_rule", {})
        if execution_rule.get("changed_from_v31") is not False \
                or execution_rule.get("required_consecutive_confirmations") != 2 \
                or execution_rule.get("target_hold_seconds") != 172800 \
                or execution_rule.get("hard_max_age_seconds") != max_age \
                or execution_rule.get("cost_or_outcome_inputs") is not False:
            raise ContractError(f"{cycle['cycle_id']} result changed the V31 execution state machine")
        if any(row.get("execution_selected") is not True for row in rows):
            raise ContractError(f"{cycle['cycle_id']} dropped a generated RAW signal from the execution ledger")
        if any(not isinstance(row.get("arm_actions"), dict)
               or set(row["arm_actions"]) != set(ARMS)
               or len(set(row["arm_actions"].values())) != 1 for row in rows):
            raise ContractError(f"{cycle['cycle_id']} execution actions differ across cost arms")
        if payload.get("signal_id_set_sha256") \
                != hashlib.sha256(canonical_bytes(sorted(ids))).hexdigest():
            raise ContractError(f"{cycle['cycle_id']} embedded signal-id set hash mismatch")
        if cycle["cycle_id"] == "V33":
            runtime = payload.get("runtime_compatibility_provenance", {})
            if runtime.get("classification") != "NON_STRATEGY_RUNTIME_COMPATIBILITY" \
                    or runtime.get("changed_strategy_variables") != 0 \
                    or runtime.get("same_unobserved_v32_strategy") is not True \
                    or runtime.get("v32_rerun_permitted") is not False \
                    or runtime.get("instant_changed") is not False:
                raise ContractError("V33 result runtime compatibility provenance mismatch")
            if any(not all(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z", row[field])
                           for field in ("decision_time", "fill_time", "exit_time")) for row in rows):
                raise ContractError("V33 ledger contains a noncanonical scheduled timestamp")

    periods = payload.get("periods", {})
    if set(periods) != set(PERIODS):
        raise ContractError("walk-forward or comparable-month set changed")
    signal_sets: dict[str, list[str]] = {arm: list(ids) for arm in ARMS}
    for period_name in PERIODS:
        period = periods[period_name]
        raw_count = period.get("raw_diagnostics", {}).get("signals")
        if not isinstance(raw_count, int):
            raise ContractError(f"missing RAW signal count in {period_name}")
        for arm in ARMS:
            metrics = period.get(arm)
            if not isinstance(metrics, dict) or metrics.get("source_signals") != raw_count:
                raise ContractError(f"signal set/count mismatch in {period_name}/{arm}")
            if raw_source == "SEALED_PARENT_V25_LEDGER" and cycle["cycle_id"] in {"V26", "V27"}:
                start, end = PERIOD_BOUNDS[period_name]
                selected_count = sum(
                    row.get("execution_selected") is True
                    and start <= row["fill_time"][:10] < end
                    and row["exit_time"][:10] < end
                    for row in rows
                )
                if metrics.get("executed_signals") != selected_count:
                    raise ContractError(f"V26/V27 executed signal count mismatch in {period_name}/{arm}")
            if cycle["cycle_id"] in {"V28", "V29", "V30", "V31"}:
                required_metrics = {
                    "gross_edge_bps", "realized_cost_bps", "net_edge_bps", "turnover_nav",
                    "break_even_cost_bps", "direction_accuracy", "equity_multiple", "max_drawdown",
                    "max_margin_requirement_jpy_at_1x", "terminal_inventory_mtm",
                    "max_inventory_age_seconds", "N_eff_days", "execution_state_transition_sha256",
                }
                if set(metrics) < required_metrics:
                    raise ContractError(f"{cycle['cycle_id']} required metrics missing in {period_name}/{arm}")
                if metrics.get("processed_raw_signals") != raw_count or metrics.get("cash_signals") != 0:
                    raise ContractError(f"{cycle['cycle_id']} did not process the complete RAW ledger in {period_name}/{arm}")
                if metrics["max_inventory_age_seconds"] > max_age:
                    raise ContractError(f"{cycle['cycle_id']} max-age exceeded in {period_name}/{arm}")
                if cycle["cycle_id"] == "V29" and not isinstance(metrics.get("basket_consensus_releases"), int):
                    raise ContractError(f"V29 consensus release count missing in {period_name}/{arm}")
                if cycle["cycle_id"] == "V30" and not isinstance(metrics.get("scope_release_count"), int):
                    raise ContractError(f"V30 scoped release count missing in {period_name}/{arm}")
                if cycle["cycle_id"] == "V31" and (
                        not isinstance(metrics.get("persistence_release_count"), int)
                        or not isinstance(metrics.get("persistence_armed_count"), int)
                        or not isinstance(metrics.get("persistence_reset_count"), int)):
                    raise ContractError(f"V31 persistence counts missing in {period_name}/{arm}")
            if cycle["cycle_id"] in {"V32", "V33"}:
                required_metrics = {
                    "gross_edge_bps", "realized_cost_bps", "net_edge_bps", "turnover_nav",
                    "break_even_cost_bps", "direction_accuracy", "equity_multiple", "max_drawdown",
                    "max_margin_requirement_jpy_at_1x", "terminal_inventory_mtm",
                    "max_inventory_age_seconds", "N_eff_days", "N_eff_episodes",
                    "max_gross_exposure_nav", "max_currency_abs_exposure_nav",
                    "execution_state_transition_sha256",
                }
                if set(metrics) < required_metrics:
                    raise ContractError(f"{cycle['cycle_id']} required metrics missing in {period_name}/{arm}")
                if metrics.get("processed_raw_signals") != raw_count or metrics.get("cash_signals") != 0:
                    raise ContractError(f"{cycle['cycle_id']} did not process the complete RAW ledger in {period_name}/{arm}")
                if metrics["max_inventory_age_seconds"] > max_age:
                    raise ContractError(f"{cycle['cycle_id']} max-age exceeded in {period_name}/{arm}")
            if metrics.get("terminal_open_inventory") != 0:
                raise ContractError(f"terminal inventory nonzero in {period_name}/{arm}")
            if not isinstance(metrics.get("equity_multiple"), int | float):
                raise ContractError(f"missing equity multiple in {period_name}/{arm}")
            if cycle["cycle_id"] in {"V27", "V28", "V29", "V30", "V31"}:
                if not isinstance(metrics.get("max_gross_exposure_nav"), int | float) \
                        or not isinstance(metrics.get("max_margin_requirement_jpy_at_1x"), int | float):
                    raise ContractError(f"missing margin metrics in {period_name}/{arm}")
                if metrics["max_gross_exposure_nav"] < 0 \
                        or metrics["max_gross_exposure_nav"] > cycle["inventory_contract"]["rule_max_gross_leverage"]:
                    raise ContractError(f"margin exposure exceeds preregistration in {period_name}/{arm}")
        if cycle["cycle_id"] in {"V28", "V29", "V30", "V31"}:
            transition_hashes = {
                period[arm]["execution_state_transition_sha256"] for arm in ARMS
            }
            if len(transition_hashes) != 1:
                raise ContractError(f"{cycle['cycle_id']} arm transitions differ in {period_name}")
        if cycle["cycle_id"] in {"V32", "V33"}:
            transition_hashes = {
                period[arm]["execution_state_transition_sha256"] for arm in ARMS
            }
            if len(transition_hashes) != 1:
                raise ContractError(f"{cycle['cycle_id']} arm transitions differ")
    if len({tuple(value) for value in signal_sets.values()}) != 1:
        raise ContractError("same signal_id set assertion failed")

    return {
        "result_file_sha256": sha256_file(result_path),
        "embedded_result_sha256": payload["result_sha256"],
        "ledger_sha256": ledger_hash,
        "signal_id_set_sha256": hashlib.sha256(canonical_bytes(sorted(ids))).hexdigest(),
        "signals": len(ids),
        "effective_bet_days": payload.get("effective_bet_days"),
        "periods": periods,
        "development_admitted": payload.get("development_admitted") is True,
        "result": payload,
    }


def evaluate_gates(registry: dict[str, Any], cycle: dict[str, Any], verified: dict[str, Any]) -> dict[str, Any]:
    gate = registry["profit_gate"]
    periods = verified["periods"]
    months = cycle["evaluation_contract"]["full_comparable_months"]
    normal = {month: periods[month]["EXECUTABLE_BASE"]["equity_multiple"] for month in months}
    adverse = {month: periods[month]["ADVERSE_STRESS"]["equity_multiple"] for month in months}
    normal_pass = all(value >= gate["normal_min_multiple"] for value in normal.values())
    adverse_pass = all(value >= gate["adverse_min_multiple"] for value in adverse.values())
    stretch = all(value >= gate["stretch_multiple"] for value in (*normal.values(), *adverse.values()))
    holdout = cycle["evaluation_contract"]["holdout"]
    holdout_reproduced = holdout["state"] == "OPENED_REPRODUCED"
    profit_pass = normal_pass and adverse_pass and holdout_reproduced
    return {
        "system_acceptance": {
            "passed": True,
            "paper_only": True,
            "external_orders": 0,
            "restart_safe_seal": True,
            "holdout_state": holdout["state"],
        },
        "strategy_profit_gate": {
            "passed": profit_pass,
            "initial_equity_jpy": gate["initial_equity_jpy"],
            "normal_monthly_multiples": normal,
            "adverse_monthly_multiples": adverse,
            "normal_2x_pass": normal_pass,
            "adverse_2x_pass": adverse_pass,
            "unopened_holdout_reproduced": holdout_reproduced,
            "stretch_3x_pass": stretch,
            "adoption_authorized": False,
        },
    }


def route_next_work_order(
    result_reason: str | None,
    raw: float,
    base: float,
    adverse: float,
    consecutive_prior_raw_edge_refinements: int,
    policy_applies: bool,
) -> tuple[str, str]:
    """Choose one successor variable without allowing unbounded exit refinements."""
    if (policy_applies and raw <= 1 and result_reason in RAW_EDGE_REFINEMENT_REASONS
            and consecutive_prior_raw_edge_refinements >= RAW_EDGE_REFINEMENT_BUDGET):
        return SIGNAL_FAMILY_PIVOT_REASON, SIGNAL_FAMILY_PIVOT_VARIABLE
    if result_reason == "EXECUTION_SUBSET_RAW_EDGE_ABSENT":
        reason = result_reason
        variable = "one_preregistered_causal_basket_hold_rule_that_preserves_all_v25_raw_signals_and_fixed_sleeves"
    elif result_reason == "BASKET_HOLD_RAW_EDGE_ABSENT":
        reason = result_reason
        variable = "one_preregistered_causal_basket_consensus_release_rule_preserving_all_v25_raw_signals_and_fixed_sleeves"
    elif result_reason == "BASKET_HOLD_RAW_EDGE_COST_DOMINANT":
        reason = result_reason
        variable = "one_preregistered_causal_hold_duration_rule_preserving_all_v25_raw_signals_and_fixed_sleeves"
    elif result_reason == "BASKET_HOLD_ADVERSE_COST_FRAGILE":
        reason = result_reason
        variable = "one_preregistered_causal_opposite_signal_release_rule_preserving_all_v25_raw_signals_and_fixed_sleeves"
    elif result_reason == "BASKET_CONSENSUS_RELEASE_RAW_EDGE_ABSENT":
        reason = result_reason
        variable = "one_preregistered_causal_consensus_release_scope_rule_preserving_all_v25_raw_signals_and_fixed_sleeves"
    elif result_reason == "BASKET_CONSENSUS_RELEASE_COST_DOMINANT":
        reason = result_reason
        variable = "one_preregistered_causal_consensus_release_turnover_rule_preserving_all_v25_raw_signals_and_fixed_sleeves"
    elif result_reason == "BASKET_CONSENSUS_RELEASE_ADVERSE_COST_FRAGILE":
        reason = result_reason
        variable = "one_preregistered_causal_consensus_release_cost_robustness_rule_preserving_all_v25_raw_signals_and_fixed_sleeves"
    elif result_reason == "CONSENSUS_RELEASE_SCOPE_RAW_EDGE_ABSENT":
        reason = result_reason
        variable = "one_preregistered_causal_consensus_release_persistence_confirmation_rule_preserving_all_v25_raw_signals_and_fixed_sleeves"
    elif result_reason == "CONSENSUS_RELEASE_SCOPE_COST_DOMINANT":
        reason = result_reason
        variable = "one_preregistered_causal_consensus_release_scope_turnover_rule_preserving_all_v25_raw_signals_and_fixed_sleeves"
    elif result_reason == "CONSENSUS_RELEASE_SCOPE_ADVERSE_COST_FRAGILE":
        reason = result_reason
        variable = "one_preregistered_causal_consensus_release_scope_cost_robustness_rule_preserving_all_v25_raw_signals_and_fixed_sleeves"
    elif result_reason == "FX_SESSION_HANDOFF_FADE_RAW_EDGE_ABSENT":
        reason = result_reason
        variable = SIGNAL_FAMILY_PIVOT_VARIABLE
    elif result_reason == "FX_SESSION_HANDOFF_FADE_COST_DOMINANT":
        reason = result_reason
        variable = "one_preregistered_turnover_reduction_rule_preserving_all_v32_raw_signals"
    elif result_reason == "FX_SESSION_HANDOFF_FADE_ADVERSE_COST_FRAGILE":
        reason = result_reason
        variable = "one_preregistered_cost_robustness_rule_preserving_all_v32_raw_signals"
    elif result_reason == "MONTHLY_2X_AND_UNOPENED_HOLDOUT_NOT_MET":
        reason = result_reason
        variable = "one_preregistered_causal_inventory_carry_rule_with_finite_max_age_and_unchanged_v25_raw_signals"
    elif raw <= 1:
        reason = "RAW_EDGE_ABSENT"
        variable = "replace_the_signal_family_without_changing_costs_or_leverage"
    elif base <= 1:
        reason = "RAW_EDGE_COST_DOMINANT"
        variable = "one_preregistered_turnover_reduction_rule_with_identical_raw_signal_definition"
    elif adverse <= 1:
        reason = "ADVERSE_COST_FRAGILE"
        variable = "one_preregistered_edge_per_decision_filter_not_using_cost_or_future_outcomes"
    else:
        reason = "DEVELOPMENT_EDGE_NOT_FINAL"
        variable = "one_preregistered_causal_regime_partition_on_development_data_only"
    return reason, variable


def sealed_raw_edge_refinement_history(
    root: Path,
    registry: dict[str, Any],
    state: dict[str, Any],
    before_cycle_id: str,
) -> list[dict[str, Any]]:
    """Derive the contiguous predecessor run from sealed state, journal and results."""
    before_number = int(before_cycle_id.removeprefix("V"))
    journal_path = state_paths(root, before_cycle_id)["journal"]
    journal = []
    if journal_path.is_file():
        journal = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    history: list[dict[str, Any]] = []
    predecessors = sorted(
        (cycle for cycle in registry["cycles"]
         if int(cycle["cycle_id"].removeprefix("V")) < before_number),
        key=lambda cycle: int(cycle["cycle_id"].removeprefix("V")),
        reverse=True,
    )
    expected_number = before_number - 1
    for predecessor in predecessors:
        cycle_id = predecessor["cycle_id"]
        number = int(cycle_id.removeprefix("V"))
        if number != expected_number:
            break
        record = state.get("cycles", {}).get(cycle_id, {})
        if not str(record.get("status", "")).startswith("SEALED_"):
            break
        result_path = within(root, predecessor["execution"]["result"])
        result_hash = record.get("official_result_file_sha256")
        if not result_path.is_file() or sha256_file(result_path) != result_hash:
            raise ContractError(f"sealed predecessor result mismatch: {cycle_id}")
        matching_events = [event for event in journal
                           if event.get("event") == "OFFICIAL_RESULT_SEALED"
                           and event.get("cycle_id") == cycle_id
                           and event.get("result_file_sha256") == result_hash]
        if len(matching_events) != 1:
            raise ContractError(f"sealed predecessor journal evidence mismatch: {cycle_id}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        reason = payload.get("automatic_rejection", {}).get("reason_code")
        if reason not in RAW_EDGE_REFINEMENT_REASONS:
            break
        history.append({
            "cycle_id": cycle_id,
            "reason_code": reason,
            "result_file_sha256": result_hash,
            "official_seal_sha256": record.get("official_seal_sha256"),
            "journal_event": "OFFICIAL_RESULT_SEALED",
        })
        expected_number -= 1
    history.reverse()
    return history


def next_work_order(
    root: Path,
    registry: dict[str, Any],
    state: dict[str, Any],
    cycle: dict[str, Any],
    verified: dict[str, Any],
) -> dict[str, Any]:
    walk = verified["periods"]["WALK_FORWARD"]
    raw = walk["RAW_SIGNAL"]["equity_multiple"]
    base = walk["EXECUTABLE_BASE"]["equity_multiple"]
    adverse = walk["ADVERSE_STRESS"]["equity_multiple"]
    result_reason = verified["result"].get("automatic_rejection", {}).get("reason_code")
    policy_applies = int(cycle["cycle_id"].removeprefix("V")) >= 31
    history = (sealed_raw_edge_refinement_history(root, registry, state, cycle["cycle_id"])
               if policy_applies else [])
    reason, variable = route_next_work_order(
        result_reason, raw, base, adverse, len(history), policy_applies,
    )
    next_number = int(cycle["cycle_id"].removeprefix("V")) + 1
    return {
        "schema_version": 1,
        "parent_cycle": cycle["cycle_id"],
        "proposed_cycle": f"V{next_number}",
        "status": "PROPOSAL_ONLY_NOT_REGISTERED_NOT_EXECUTABLE",
        "reason_code": reason,
        "single_next_changed_variable": variable,
        "constraints": {
            "one_variable_only": True,
            "same_raw_signal_ids_across_cost_arms": True,
            "holdout_must_remain_unopened": True,
            "leverage_may_not_be_changed_after_results": True,
            "evaluation_period_may_not_be_tuned": True,
        },
        "observed_walk_forward": {"RAW_SIGNAL": raw, "EXECUTABLE_BASE": base, "ADVERSE_STRESS": adverse},
        "raw_edge_refinement_budget_policy": {
            "classification": "NON_STRATEGY_ORCHESTRATOR_POLICY",
            "effective": policy_applies,
            "max_consecutive_refinements": RAW_EDGE_REFINEMENT_BUDGET,
            "consecutive_prior_refinements": len(history),
            "derived_from_sealed_cycles": history,
            "budget_reached": policy_applies and len(history) >= RAW_EDGE_REFINEMENT_BUDGET,
            "policy_path": RAW_EDGE_REFINEMENT_POLICY_PATH,
        },
        "authority": AUTHORITY,
    }


def state_paths(root: Path, cycle_id: str = "V25") -> dict[str, Path]:
    base = within(root, "evidence/orchestrator_state_v2")
    try:
        next_number = int(cycle_id.removeprefix("V")) + 1
    except ValueError as error:
        raise ContractError(f"invalid cycle id: {cycle_id}") from error
    return {
        "base": base,
        "state": base / "state.json",
        "journal": base / "failure_and_event_journal.jsonl",
        "lock": base / "orchestrator.lock",
        "seal": base / f"official_seal_{cycle_id.lower()}.json",
        "work_order": base / f"next_hypothesis_work_order_v{next_number}.json",
    }


def read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 2, "cycles": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def exclusive_lock(path: Path, journal: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        try:
            stale_pid = int(path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            stale_pid = -1
        if stale_pid > 0 and pid_alive(stale_pid):
            raise ContractError(f"orchestrator is already active with pid {stale_pid}")
        append_journal(journal, "STALE_LOCK_RECOVERED", stale_pid=stale_pid)
        path.unlink()
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


def seal_completed(root: Path, registry: dict[str, Any], cycle: dict[str, Any], state: dict[str, Any],
                   paths: dict[str, Path], recovery: bool,
                   authorized_recovery: bool = False) -> dict[str, Any]:
    verified = validate_result(root, cycle)
    gates = evaluate_gates(registry, cycle, verified)
    seal = {
        "schema_version": 2,
        "cycle_id": cycle["cycle_id"],
        "official_execution_ordinal": 1,
        "recovered_without_rerun": recovery,
        "authorized_recovery_execution": authorized_recovery,
        "authorized_recovery_ordinal": 1 if authorized_recovery else 0,
        "recovery_authorization_sha256": (
            V26_RECOVERY_AUTHORIZATION_SHA256 if authorized_recovery else None
        ),
        "registry_sha256": sha256_file(within(root, "PAPER_RESEARCH_CYCLE_REGISTRY_V2.json")),
        "preregistration_sha256": cycle["preregistration_sha256"],
        "script_sha256": cycle["script_sha256"],
        "test_sha256": cycle["test_sha256"],
        "source_manifest_sha256": cycle["source_contract"]["manifest_sha256"],
        "result_file_sha256": verified["result_file_sha256"],
        "embedded_result_sha256": verified["embedded_result_sha256"],
        "ledger_sha256": verified["ledger_sha256"],
        "signal_id_set_sha256": verified["signal_id_set_sha256"],
        "signals": verified["signals"],
        "effective_bet_days": verified["effective_bet_days"],
        **gates,
        "authority": AUTHORITY,
    }
    seal["official_seal_sha256"] = embedded_hash(seal, "official_seal_sha256")
    atomic_json(paths["seal"], seal)
    atomic_json(paths["work_order"], next_work_order(root, registry, state, cycle, verified))
    state["cycles"][cycle["cycle_id"]].update({
        "status": "SEALED_SYSTEM_PASS_PROFIT_UNPROVEN" if not gates["strategy_profit_gate"]["passed"]
        else "SEALED_SYSTEM_AND_PROFIT_PASS_NOT_ADOPTED",
        "official_result_file_sha256": verified["result_file_sha256"],
        "official_seal_file_sha256": sha256_file(paths["seal"]),
        "official_seal_sha256": seal["official_seal_sha256"],
        "strategy_profit_pass": gates["strategy_profit_gate"]["passed"],
    })
    atomic_json(paths["state"], state)
    append_journal(paths["journal"], "OFFICIAL_RESULT_SEALED", cycle_id=cycle["cycle_id"],
                   result_file_sha256=verified["result_file_sha256"],
                   official_seal_sha256=seal["official_seal_sha256"], recovery=recovery,
                   authorized_recovery=authorized_recovery,
                   system_pass=True, profit_pass=gates["strategy_profit_gate"]["passed"])
    return seal


def validate_v26_recovery_work_order(root: Path, cycle: dict[str, Any],
                                     cycle_state: dict[str, Any]) -> dict[str, Any]:
    """Validate the non-executable, pre-result V26 recovery proposal."""
    path = within(root, V26_RECOVERY_WORK_ORDER)
    if not path.is_file() or sha256_file(path) != V26_RECOVERY_WORK_ORDER_SHA256:
        raise ContractError("V26 recovery work order is missing or changed")
    work_order = json.loads(path.read_text(encoding="utf-8"))
    if work_order.get("cycle_id") != "V26" or work_order.get("authority") != AUTHORITY:
        raise ContractError("V26 recovery work order identity or authority mismatch")
    failure = work_order.get("failure_evidence", {})
    if cycle_state.get("status") != "FAILED_OFFICIAL_EXECUTION_NO_RERUN" \
            or cycle_state.get("official_attempts") != 1:
        raise ContractError("V26 recovery work order is not bound to the terminal failed attempt")
    if failure.get("official_attempts") != 1 or failure.get("result_file_exists") is not False \
            or failure.get("ledger_file_exists") is not False:
        raise ContractError("V26 recovery work order does not preserve the pre-result failure")
    if failure.get("persisted_or_reported_strategy_metrics_observed") is not False \
            or failure.get("diagnostic_replay_result_reusable") is not False:
        raise ContractError("V26 recovery work order permits outcome leakage or diagnostic reuse")
    if failure.get("stdout_sha256") != cycle_state.get("stdout_sha256") \
            or failure.get("stderr_sha256") != cycle_state.get("stderr_sha256"):
        raise ContractError("V26 recovery failure evidence hashes differ from state")

    execution = cycle["execution"]
    if within(root, execution["result"]).exists() or within(root, execution["ledger"]).exists():
        raise ContractError("V26 recovery proposal cannot coexist with an unsealed result or ledger")
    frozen = work_order.get("frozen_strategy_contract", {})
    expected_frozen = {
        "preregistration_sha256": cycle["preregistration_sha256"],
        "original_runner_sha256": cycle["script_sha256"],
        "original_test_sha256": cycle["test_sha256"],
        "source_manifest_sha256": cycle["source_contract"]["manifest_sha256"],
        "parent_ledger_sha256": cycle["signal_contract"]["parent_ledger_sha256"],
        "parent_signal_id_set_sha256": cycle["signal_contract"]["parent_signal_id_set_sha256"],
    }
    if any(frozen.get(key) != value for key, value in expected_frozen.items()):
        raise ContractError("V26 recovery work order changed a frozen strategy input")
    parent_rows = [json.loads(line) for line in within(
        root, cycle["signal_contract"]["parent_ledger"]
    ).read_text(encoding="utf-8").splitlines() if line]
    timestamp_values = [
        row[field]
        for row in parent_rows
        for field in ("decision_time", "fill_time", "exit_time")
    ]
    timestamp_evidence = work_order.get("timestamp_evidence", {})
    if timestamp_evidence.get("parent_rows") != len(parent_rows) \
            or timestamp_evidence.get("timestamps_checked") != len(timestamp_values) \
            or timestamp_evidence.get("nonconforming_actual_timestamps") != 0 \
            or any(not value.endswith(".000000000Z") for value in timestamp_values):
        raise ContractError("V26 recovery timestamp evidence differs from the frozen parent ledger")

    repair = work_order.get("repair_contract", {})
    if repair.get("allowed_changed_variable_count") != 1 \
            or repair.get("allowed_change") != "Python_3_10_timestamp_parser_compatibility_only":
        raise ContractError("V26 recovery repair scope is not timestamp-only")
    for path_key, hash_key in (
        ("compatibility_module", "compatibility_module_sha256"),
        ("compatibility_test", "compatibility_test_sha256"),
    ):
        artifact = within(root, repair.get(path_key, ""))
        if not artifact.is_file() or sha256_file(artifact) != repair.get(hash_key):
            raise ContractError(f"V26 recovery artifact changed: {path_key}")
    authorization = work_order.get("authorization_gate", {})
    if authorization.get("explicit_user_authorization_required") is not True \
            or authorization.get("authorization_recorded") is not False \
            or authorization.get("current_launcher_registered") is not False \
            or authorization.get("current_execution_allowed") is not False:
        raise ContractError("V26 recovery authorization gate is not closed")
    return {
        "status": work_order["status"],
        "reason_code": work_order["reason_code"],
        "work_order_sha256": V26_RECOVERY_WORK_ORDER_SHA256,
        "authorization_recorded": False,
        "execution_allowed": False,
    }


def validate_v26_recovery_authorization(root: Path, cycle: dict[str, Any],
                                        cycle_state: dict[str, Any]) -> dict[str, Any]:
    """Bind the user's one-shot permission to the frozen timestamp-only repair."""
    path = within(root, V26_RECOVERY_AUTHORIZATION)
    if not path.is_file() or sha256_file(path) != V26_RECOVERY_AUTHORIZATION_SHA256:
        raise ContractError("V26 recovery authorization is missing or changed")
    authorization = json.loads(path.read_text(encoding="utf-8"))
    if authorization.get("schema_version") != 1 or authorization.get("cycle_id") != "V26" \
            or authorization.get("authorized") is not True:
        raise ContractError("V26 recovery authorization identity is invalid")
    if authorization.get("scope") != "ONE_TIMESTAMP_ONLY_PAPER_RECOVERY_ATTEMPT" \
            or authorization.get("recovery_attempt_limit") != 1:
        raise ContractError("V26 recovery authorization is not one-shot/timestamp-only")
    source = authorization.get("authorization_source", {})
    if source.get("owner_task_id") != "01a03f46-9dfd-7042-bfd1-d3ba26072171" \
            or source.get("user_message_exact") != "許可" \
            or source.get("user_message_utf8_sha256") != hashlib.sha256("許可".encode()).hexdigest():
        raise ContractError("V26 recovery user authorization binding changed")
    if authorization.get("authority") != AUTHORITY:
        raise ContractError("V26 recovery authorization exceeds paper-only authority")

    failure = authorization.get("pre_result_failure_binding", {})
    if failure.get("state") != "FAILED_OFFICIAL_EXECUTION_NO_RERUN" \
            or failure.get("official_attempts") != 1 \
            or failure.get("stdout_sha256") != cycle_state.get("stdout_sha256") \
            or failure.get("stderr_sha256") != cycle_state.get("stderr_sha256"):
        raise ContractError("V26 recovery authorization is not bound to the failed attempt")
    if failure.get("result_file_exists") is not False or failure.get("ledger_file_exists") is not False:
        raise ContractError("V26 recovery authorization is not pre-result")

    frozen = authorization.get("frozen_strategy_hashes", {})
    expected_frozen = {
        "preregistration_sha256": cycle["preregistration_sha256"],
        "original_runner_sha256": cycle["script_sha256"],
        "original_test_sha256": cycle["test_sha256"],
        "parent_ledger_sha256": cycle["signal_contract"]["parent_ledger_sha256"],
        "parent_signal_id_set_sha256": cycle["signal_contract"]["parent_signal_id_set_sha256"],
    }
    if any(frozen.get(key) != value for key, value in expected_frozen.items()):
        raise ContractError("V26 recovery authorization changed a frozen strategy hash")
    repair = authorization.get("repair_hashes", {})
    expected_repair = {
        "work_order_sha256": V26_RECOVERY_WORK_ORDER_SHA256,
        "compatibility_module_sha256": "f1b68055a77664e7a33ab9dc04ef068de152f4712c9f0bf81ce929d141886585",
        "compatibility_test_sha256": "cb26f827885b5ef82f98d886511b2cc783ff23427b198002714f2e3113ff2f5b",
    }
    if any(repair.get(key) != value for key, value in expected_repair.items()):
        raise ContractError("V26 recovery authorization changed the timestamp-only repair")
    launcher = within(root, authorization.get("one_shot_launcher", ""))
    if authorization.get("one_shot_launcher") != V26_RECOVERY_LAUNCHER \
            or not launcher.is_file() \
            or authorization.get("one_shot_launcher_sha256") != V26_RECOVERY_LAUNCHER_SHA256 \
            or sha256_file(launcher) != V26_RECOVERY_LAUNCHER_SHA256:
        raise ContractError("V26 recovery one-shot launcher is missing or changed")
    return {
        "status": "AUTHORIZED_ONE_SHOT_RECOVERY_PENDING",
        "authorization_sha256": V26_RECOVERY_AUTHORIZATION_SHA256,
        "authorization_recorded": True,
        "execution_allowed": cycle_state.get("status") == "FAILED_OFFICIAL_EXECUTION_NO_RERUN",
        "recovery_attempt_limit": 1,
    }


def validate_v26_recovery_failure(root: Path, cycle: dict[str, Any],
                                  cycle_state: dict[str, Any]) -> dict[str, Any]:
    """Validate the terminal no-result evidence without inventing strategy metrics."""
    path = within(root, V26_RECOVERY_FAILURE)
    if not path.is_file() or sha256_file(path) != V26_RECOVERY_FAILURE_SHA256:
        raise ContractError("V26 authorized recovery failure evidence is missing or changed")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("cycle_id") != "V26" \
            or evidence.get("status") != "FAILED_AUTHORIZED_RECOVERY_NO_RESULT_RERUN_FORBIDDEN" \
            or evidence.get("authority") != AUTHORITY:
        raise ContractError("V26 authorized recovery failure identity or authority changed")
    execution = evidence.get("execution_evidence", {})
    if cycle_state.get("status") != "FAILED_AUTHORIZED_RECOVERY_NO_RERUN" \
            or cycle_state.get("official_attempts") != 1 \
            or cycle_state.get("recovery_attempts") != 1 \
            or execution.get("subprocess_returncode") != cycle_state.get("recovery_subprocess_returncode") \
            or execution.get("stdout_sha256") != cycle_state.get("recovery_stdout_sha256") \
            or execution.get("stderr_sha256") != cycle_state.get("recovery_stderr_sha256"):
        raise ContractError("V26 terminal recovery evidence differs from state")
    if execution.get("result_file_exists") is not False \
            or execution.get("ledger_file_exists") is not False \
            or execution.get("second_recovery_forbidden") is not True:
        raise ContractError("V26 terminal recovery evidence permits output reuse or rerun")
    if within(root, cycle["execution"]["result"]).exists() \
            or within(root, cycle["execution"]["ledger"]).exists():
        raise ContractError("V26 terminal recovery evidence conflicts with an output file")
    binding = evidence.get("binding", {})
    if binding.get("authorization_sha256") != V26_RECOVERY_AUTHORIZATION_SHA256 \
            or binding.get("launcher_sha256") != V26_RECOVERY_LAUNCHER_SHA256 \
            or binding.get("preregistration_sha256") != cycle["preregistration_sha256"] \
            or binding.get("original_runner_sha256") != cycle["script_sha256"] \
            or binding.get("original_test_sha256") != cycle["test_sha256"]:
        raise ContractError("V26 terminal recovery evidence changed a frozen binding")
    classification = evidence.get("cause_classification", {})
    if classification.get("exact_traceback_available") is not False \
            or classification.get("classification_strength") \
            != "INFERRED_FROM_STATIC_CALL_PATH_AND_BOUNDED_SINGLE_FUNCTION_REPRODUCTION" \
            or classification.get("bounded_reproduction", {}).get("strategy_replay_performed") is not False:
        raise ContractError("V26 failure classification overstates the available evidence")
    strategy = evidence.get("strategy_evidence", {})
    unavailable = (
        "gross_edge", "realized_cost", "net_edge", "turnover", "break_even_cost",
        "direction_accuracy", "monthly_multiples", "max_drawdown", "terminal_mtm",
        "inventory_age", "N_eff",
    )
    if strategy.get("result_observed") is not False \
            or strategy.get("metrics_available") is not False \
            or strategy.get("profit_proven") is not False \
            or strategy.get("strategy_adoption_authorized") is not False \
            or any(strategy.get(field) is not None for field in unavailable):
        raise ContractError("V26 no-result failure fabricates strategy evidence")
    work_order = evidence.get("next_work_order", {})
    if work_order.get("status") != "ENGINEERING_FORENSIC_ONLY_NOT_A_STRATEGY_HYPOTHESIS_NOT_EXECUTABLE" \
            or work_order.get("v26_may_not_be_replayed") is not True \
            or work_order.get("future_strategy_cycle_requires_new_preregistration") is not True:
        raise ContractError("V26 terminal work order is executable or permits a replay")
    return {
        "authorization_recorded": True,
        "authorization_sha256": V26_RECOVERY_AUTHORIZATION_SHA256,
        "execution_allowed": False,
        "recovery_attempts": 1,
        "failure_evidence_sha256": V26_RECOVERY_FAILURE_SHA256,
        "metrics_available": False,
        "profit_proven": False,
        "next_work_order": work_order,
    }


def audit(root: Path, registry: dict[str, Any]) -> dict[str, Any]:
    shared_paths = state_paths(root)
    state = read_state(shared_paths["state"])
    reports = []
    for cycle in registry["cycles"]:
        paths = state_paths(root, cycle["cycle_id"])
        source_audit = validate_source(cycle)
        cycle_state = state.get("cycles", {}).get(cycle["cycle_id"])
        result_path = within(root, cycle["execution"]["result"])
        if cycle_state is None and result_path.exists():
            raise ContractError("official-looking result exists without an execution intent")
        if cycle_state and cycle_state.get("status", "").startswith("SEALED"):
            verified = validate_result(root, cycle)
            if verified["result_file_sha256"] != cycle_state.get("official_result_file_sha256"):
                raise ContractError("sealed official result changed")
            if not paths["seal"].is_file() or sha256_file(paths["seal"]) != cycle_state.get("official_seal_file_sha256"):
                raise ContractError("official seal file changed")
            status = cycle_state["status"]
        elif cycle_state and cycle_state.get("status") == "ATTEMPT_STARTED" and result_path.exists():
            status = "RECOVERABLE_RESULT_NOT_YET_SEALED"
        elif cycle_state and cycle_state.get("status") == "ATTEMPT_STARTED":
            status = "FAIL_CLOSED_UNCERTAIN_EXECUTION_NO_RESULT"
        elif cycle_state and cycle_state.get("status") == "FAILED_OFFICIAL_EXECUTION_NO_RERUN":
            status = "AUTHORIZED_ONE_SHOT_RECOVERY_PENDING" if cycle["cycle_id"] == "V26" \
                else "FAILED_OFFICIAL_EXECUTION_NO_RESULT_RERUN_FORBIDDEN"
            if cycle["cycle_id"] == "V26":
                validate_v26_recovery_work_order(root, cycle, cycle_state)
                recovery = validate_v26_recovery_authorization(root, cycle, cycle_state)
            else:
                recovery = None
        elif cycle_state and cycle_state.get("status") == "RECOVERY_ATTEMPT_STARTED":
            result_exists = result_path.exists()
            ledger_exists = within(root, cycle["execution"]["ledger"]).exists()
            status = "RECOVERABLE_AUTHORIZED_RESULT_NOT_YET_SEALED" \
                if result_exists and ledger_exists else "FAIL_CLOSED_UNCERTAIN_RECOVERY_NO_RESULT"
            recovery = {
                "authorization_recorded": True,
                "execution_allowed": False,
                "authorization_sha256": cycle_state.get("recovery_authorization_sha256"),
                "recovery_attempts": cycle_state.get("recovery_attempts"),
            }
        elif cycle_state and cycle_state.get("status") == "FAILED_AUTHORIZED_RECOVERY_NO_RERUN":
            status = "FAILED_AUTHORIZED_RECOVERY_NO_RESULT_RERUN_FORBIDDEN"
            recovery = validate_v26_recovery_failure(root, cycle, cycle_state)
        else:
            status = "REGISTERED_PREFLIGHT_PASS_PENDING"
        report = {"cycle_id": cycle["cycle_id"], "status": status,
                  "source_rows": {pair: item["rows"] for pair, item in source_audit.items()}}
        if cycle_state and cycle["cycle_id"] == "V26" and cycle_state.get("status") in {
            "FAILED_OFFICIAL_EXECUTION_NO_RERUN", "RECOVERY_ATTEMPT_STARTED",
            "FAILED_AUTHORIZED_RECOVERY_NO_RERUN",
        }:
            report["recovery"] = recovery
        reports.append(report)
    return {"schema_version": 2, "authority": AUTHORITY, "cycles": reports}


def execute_next(root: Path, registry: dict[str, Any]) -> dict[str, Any]:
    shared_paths = state_paths(root)
    with exclusive_lock(shared_paths["lock"], shared_paths["journal"]):
        state = read_state(shared_paths["state"])
        cycle = next((item for item in registry["cycles"]
                      if not state.get("cycles", {}).get(item["cycle_id"], {}).get("status", "").startswith("SEALED")
                      and state.get("cycles", {}).get(item["cycle_id"], {}).get("status")
                      not in TERMINAL_NO_RERUN_STATUSES), None)
        if cycle is None:
            raise ContractError("every registered cycle already has its one official sealed execution")
        cycle_id = cycle["cycle_id"]
        paths = state_paths(root, cycle_id)
        current = state["cycles"].get(cycle_id)
        parent_cycle = cycle.get("depends_on_cycle")
        if parent_cycle:
            parent = state.get("cycles", {}).get(parent_cycle, {})
            if not parent.get("status", "").startswith("SEALED"):
                raise ContractError(f"parent cycle {parent_cycle} is not sealed")
        result_path = within(root, cycle["execution"]["result"])
        if current and current.get("status", "").startswith("SEALED"):
            raise ContractError(f"{cycle_id} already has its one official sealed execution")
        if current and current.get("status") == "ATTEMPT_STARTED":
            if result_path.exists():
                return seal_completed(root, registry, cycle, state, paths, recovery=True)
            append_journal(paths["journal"], "FAIL_CLOSED_UNCERTAIN_EXECUTION_NO_RESULT",
                           cycle_id=cycle_id, official_attempts=current.get("official_attempts"))
            raise ContractError("an official subprocess started but no recoverable result exists; rerun forbidden")
        if current is not None:
            raise ContractError(f"unexpected cycle state: {current.get('status')}")

        source_audit = validate_source(cycle)
        state["cycles"][cycle_id] = {
            "status": "ATTEMPT_STARTED",
            "official_attempts": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "source_manifest_sha256": cycle["source_contract"]["manifest_sha256"],
        }
        atomic_json(paths["state"], state)
        append_journal(paths["journal"], "OFFICIAL_EXECUTION_STARTED", cycle_id=cycle_id,
                       official_attempt=1, source_rows={pair: item["rows"] for pair, item in source_audit.items()})
        execution = cycle["execution"]
        argv = [sys.executable, *execution["argv"]]
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(str(within(root, item)) for item in execution["pythonpath"]),
        }
        completed = subprocess.run(argv, cwd=root, env=environment, text=True, capture_output=True,
                                   timeout=execution["timeout_seconds"], check=False)
        state["cycles"][cycle_id]["subprocess_returncode"] = completed.returncode
        state["cycles"][cycle_id]["stdout_sha256"] = hashlib.sha256(completed.stdout.encode()).hexdigest()
        state["cycles"][cycle_id]["stderr_sha256"] = hashlib.sha256(completed.stderr.encode()).hexdigest()
        atomic_json(paths["state"], state)
        if completed.returncode != 0:
            excerpt = sanitized_subprocess_excerpt(completed.stderr)
            state["cycles"][cycle_id]["sanitized_stderr_excerpt"] = excerpt
            state["cycles"][cycle_id]["status"] = "FAILED_OFFICIAL_EXECUTION_NO_RERUN"
            atomic_json(paths["state"], state)
            append_journal(paths["journal"], "OFFICIAL_EXECUTION_FAILED", cycle_id=cycle_id,
                           returncode=completed.returncode,
                           stdout_sha256=state["cycles"][cycle_id]["stdout_sha256"],
                           stderr_sha256=state["cycles"][cycle_id]["stderr_sha256"],
                           sanitized_stderr_excerpt=excerpt)
            raise ContractError(f"official subprocess failed with exit {completed.returncode}; rerun forbidden")
        return seal_completed(root, registry, cycle, state, paths, recovery=False)


def execute_v26_recovery(root: Path, registry: dict[str, Any]) -> dict[str, Any]:
    """Perform exactly one explicitly authorized, timestamp-only V26 recovery."""
    shared_paths = state_paths(root)
    with exclusive_lock(shared_paths["lock"], shared_paths["journal"]):
        state = read_state(shared_paths["state"])
        cycle = next((item for item in registry["cycles"] if item["cycle_id"] == "V26"), None)
        if cycle is None:
            raise ContractError("registered V26 cycle is missing")
        paths = state_paths(root, "V26")
        current = state.get("cycles", {}).get("V26", {})
        result_path = within(root, cycle["execution"]["result"])
        ledger_path = within(root, cycle["execution"]["ledger"])

        if current.get("status", "").startswith("SEALED"):
            raise ContractError("V26 already has a sealed result; recovery rerun forbidden")
        if current.get("status") == "RECOVERY_ATTEMPT_STARTED":
            if current.get("recovery_attempts") != 1 \
                    or current.get("recovery_authorization_sha256") != V26_RECOVERY_AUTHORIZATION_SHA256:
                raise ContractError("V26 recovery state binding changed")
            if result_path.exists() and ledger_path.exists():
                return seal_completed(root, registry, cycle, state, paths, recovery=True,
                                      authorized_recovery=True)
            append_journal(paths["journal"], "FAIL_CLOSED_UNCERTAIN_AUTHORIZED_RECOVERY_NO_RESULT",
                           cycle_id="V26", recovery_attempts=current.get("recovery_attempts"))
            raise ContractError("authorized V26 recovery started without a recoverable result; rerun forbidden")
        if current.get("status") == "FAILED_AUTHORIZED_RECOVERY_NO_RERUN":
            raise ContractError("authorized V26 recovery already failed; rerun forbidden")
        if current.get("status") != "FAILED_OFFICIAL_EXECUTION_NO_RERUN" \
                or current.get("official_attempts") != 1:
            raise ContractError("V26 is not in the exact pre-result failed state")

        validate_v26_recovery_work_order(root, cycle, current)
        authorization = validate_v26_recovery_authorization(root, cycle, current)
        if result_path.exists() or ledger_path.exists():
            raise ContractError("V26 recovery outputs exist before recovery intent")
        source_audit = validate_source(cycle)
        current.update({
            "status": "RECOVERY_ATTEMPT_STARTED",
            "recovery_attempts": 1,
            "recovery_started_at": datetime.now(timezone.utc).isoformat(),
            "recovery_authorization_sha256": authorization["authorization_sha256"],
            "recovery_launcher_sha256": V26_RECOVERY_LAUNCHER_SHA256,
        })
        atomic_json(paths["state"], state)
        append_journal(paths["journal"], "AUTHORIZED_RECOVERY_EXECUTION_STARTED", cycle_id="V26",
                       official_attempts=1, recovery_attempt=1,
                       authorization_sha256=authorization["authorization_sha256"],
                       launcher_sha256=V26_RECOVERY_LAUNCHER_SHA256,
                       source_rows={pair: item["rows"] for pair, item in source_audit.items()})

        execution = cycle["execution"]
        argv = [sys.executable, V26_RECOVERY_LAUNCHER, *execution["argv"][1:]]
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(str(within(root, item)) for item in execution["pythonpath"]),
        }
        completed = subprocess.run(argv, cwd=root, env=environment, text=True, capture_output=True,
                                   timeout=execution["timeout_seconds"], check=False)
        current["recovery_subprocess_returncode"] = completed.returncode
        current["recovery_stdout_sha256"] = hashlib.sha256(completed.stdout.encode()).hexdigest()
        current["recovery_stderr_sha256"] = hashlib.sha256(completed.stderr.encode()).hexdigest()
        atomic_json(paths["state"], state)
        if completed.returncode != 0:
            current["status"] = "FAILED_AUTHORIZED_RECOVERY_NO_RERUN"
            atomic_json(paths["state"], state)
            append_journal(paths["journal"], "AUTHORIZED_RECOVERY_EXECUTION_FAILED", cycle_id="V26",
                           recovery_attempt=1, returncode=completed.returncode,
                           stdout_sha256=current["recovery_stdout_sha256"],
                           stderr_sha256=current["recovery_stderr_sha256"])
            raise ContractError(
                f"authorized V26 recovery failed with exit {completed.returncode}; rerun forbidden"
            )
        return seal_completed(root, registry, cycle, state, paths, recovery=False,
                              authorized_recovery=True)


def record_migration_journal(root: Path) -> None:
    paths = state_paths(root)
    if paths["journal"].exists() and "HANDOFF_HASH_DEFINITION_CORRECTED" in paths["journal"].read_text(encoding="utf-8"):
        return
    append_journal(paths["journal"], "HANDOFF_HASH_DEFINITION_CORRECTED",
                   obsolete_parent_prefixed_hash="20ae6fdff4a43473bc5d4004e9674e460011302f2a69fc62f4b02ed7ce49e245",
                   adopted_root_relative_hash="72834f633eb66845811165967dcb5ef42df564b621d446d28c492dba363882fa",
                   files=186, bytes=830151581,
                   explanation="same content; shasum stream path prefix differed")
    append_journal(paths["journal"], "LEGACY_SCHEMA_MIGRATION",
                   cycles="V4-V24", policy="read-only evidence; no result or seal rewrite",
                   known_fixture="V4 nanosecond timestamp fixture remains a separate migration failure")
    append_journal(paths["journal"], "V25_DIAGNOSTIC_NOT_REUSED_AS_OFFICIAL_SEAL",
                   raw_signals=500, effective_days=80,
                   walk_forward={"RAW_SIGNAL": 1.004741490752261,
                                 "EXECUTABLE_BASE": 0.9969664235923138,
                                 "ADVERSE_STRESS": 0.9907639221545058},
                   development_admitted=False, external_orders=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "execute-next", "execute-v26-recovery", "status"))
    parser.add_argument("--registry", type=Path, default=Path("PAPER_RESEARCH_CYCLE_REGISTRY_V2.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        registry = load_registry(root, args.registry)
        record_migration_journal(root)
        if args.command in {"audit", "status"}:
            result = audit(root, registry)
        elif args.command == "execute-v26-recovery":
            result = execute_v26_recovery(root, registry)
        else:
            result = execute_next(root, registry)
    except (ContractError, OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"ok": False, "error": str(error), "authority": AUTHORITY}, sort_keys=True))
        return 2
    print(json.dumps({"ok": True, "result": result}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
