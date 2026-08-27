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
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
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
    cycles = registry.get("cycles")
    if not isinstance(cycles, list) or not cycles:
        raise ContractError("registry has no cycles")
    ids = [cycle.get("cycle_id") for cycle in cycles]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ContractError("cycle ids must be present and unique")
    for cycle in cycles:
        validate_cycle_contract(root, cycle)
    return registry


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
        simulator_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Name) and node.func.id == "simulate_portfolio"]
        if len(simulator_calls) != 1 or len(simulator_calls[0].args) < 3:
            raise ContractError("cost-arm simulation call is not structurally unique")
        if not isinstance(simulator_calls[0].args[1], ast.Name) or simulator_calls[0].args[1].id != "rows":
            raise ContractError("cost arms do not consume the same RAW_SIGNAL ledger")
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
        selectors = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "apply_rule"]
        scorers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "causal_score"]
        if len(selectors) != 1 or [arg.arg for arg in selectors[0].args.args] != ["parent_rows", "corpus"]:
            raise ContractError("V26 deterministic execution selector contract mismatch")
        if len(scorers) != 1 or [arg.arg for arg in scorers[0].args.args] != ["row", "bars", "time_index"]:
            raise ContractError("V26 causal cost score contract mismatch")
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


def parse_time(stamp: str) -> datetime:
    if not stamp.endswith("Z"):
        raise ContractError(f"timestamp is not explicit UTC: {stamp}")
    head = stamp[:19]
    try:
        return datetime.fromisoformat(head).replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise ContractError(f"invalid timestamp: {stamp}") from error


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
            raise ContractError("V26 changed the V25 RAW signal identity or decision timestamps")
        if payload.get("same_execution_mask_all_cost_arms") is not True:
            raise ContractError("V26 cost arms do not share one execution mask")
        mask = []
        for row in rows:
            actions = row.get("arm_actions")
            if not isinstance(actions, dict) or set(actions) != set(ARMS) \
                    or len(set(actions.values())) != 1:
                raise ContractError("V26 ledger arm execution actions differ")
            mask.append([row["signal_id"], row.get("execution_selected") is True])
        if hashlib.sha256(canonical_bytes(mask)).hexdigest() != payload.get("execution_mask_sha256"):
            raise ContractError("V26 execution mask hash mismatch")

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
            if raw_source == "SEALED_PARENT_V25_LEDGER":
                start, end = PERIOD_BOUNDS[period_name]
                selected_count = sum(
                    row.get("execution_selected") is True
                    and start <= row["fill_time"][:10] < end
                    and row["exit_time"][:10] < end
                    for row in rows
                )
                if metrics.get("executed_signals") != selected_count:
                    raise ContractError(f"V26 executed signal count mismatch in {period_name}/{arm}")
            if metrics.get("terminal_open_inventory") != 0:
                raise ContractError(f"terminal inventory nonzero in {period_name}/{arm}")
            if not isinstance(metrics.get("equity_multiple"), int | float):
                raise ContractError(f"missing equity multiple in {period_name}/{arm}")
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


def next_work_order(cycle: dict[str, Any], verified: dict[str, Any]) -> dict[str, Any]:
    walk = verified["periods"]["WALK_FORWARD"]
    raw = walk["RAW_SIGNAL"]["equity_multiple"]
    base = walk["EXECUTABLE_BASE"]["equity_multiple"]
    adverse = walk["ADVERSE_STRESS"]["equity_multiple"]
    result_reason = verified["result"].get("automatic_rejection", {}).get("reason_code")
    if result_reason == "EXECUTION_SUBSET_RAW_EDGE_ABSENT":
        reason = result_reason
        variable = "one_preregistered_causal_basket_hold_rule_that_preserves_all_v25_raw_signals_and_fixed_sleeves"
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
                   paths: dict[str, Path], recovery: bool) -> dict[str, Any]:
    verified = validate_result(root, cycle)
    gates = evaluate_gates(registry, cycle, verified)
    seal = {
        "schema_version": 2,
        "cycle_id": cycle["cycle_id"],
        "official_execution_ordinal": 1,
        "recovered_without_rerun": recovery,
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
    atomic_json(paths["work_order"], next_work_order(cycle, verified))
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
            status = "FAILED_OFFICIAL_EXECUTION_NO_RESULT_RERUN_FORBIDDEN"
            recovery = validate_v26_recovery_work_order(root, cycle, cycle_state) \
                if cycle["cycle_id"] == "V26" else None
        else:
            status = "REGISTERED_PREFLIGHT_PASS_PENDING"
        report = {"cycle_id": cycle["cycle_id"], "status": status,
                  "source_rows": {pair: item["rows"] for pair, item in source_audit.items()}}
        if cycle_state and cycle_state.get("status") == "FAILED_OFFICIAL_EXECUTION_NO_RERUN" \
                and cycle["cycle_id"] == "V26":
            report["recovery"] = recovery
        reports.append(report)
    return {"schema_version": 2, "authority": AUTHORITY, "cycles": reports}


def execute_next(root: Path, registry: dict[str, Any]) -> dict[str, Any]:
    shared_paths = state_paths(root)
    with exclusive_lock(shared_paths["lock"], shared_paths["journal"]):
        state = read_state(shared_paths["state"])
        cycle = next((item for item in registry["cycles"]
                      if not state.get("cycles", {}).get(item["cycle_id"], {}).get("status", "").startswith("SEALED")), None)
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
            state["cycles"][cycle_id]["status"] = "FAILED_OFFICIAL_EXECUTION_NO_RERUN"
            atomic_json(paths["state"], state)
            append_journal(paths["journal"], "OFFICIAL_EXECUTION_FAILED", cycle_id=cycle_id,
                           returncode=completed.returncode,
                           stdout_sha256=state["cycles"][cycle_id]["stdout_sha256"],
                           stderr_sha256=state["cycles"][cycle_id]["stderr_sha256"])
            raise ContractError(f"official subprocess failed with exit {completed.returncode}; rerun forbidden")
        return seal_completed(root, registry, cycle, state, paths, recovery=False)


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
    parser.add_argument("command", choices=("audit", "execute-next", "status"))
    parser.add_argument("--registry", type=Path, default=Path("PAPER_RESEARCH_CYCLE_REGISTRY_V2.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        registry = load_registry(root, args.registry)
        record_migration_journal(root)
        if args.command in {"audit", "status"}:
            result = audit(root, registry)
        else:
            result = execute_next(root, registry)
    except (ContractError, OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"ok": False, "error": str(error), "authority": AUTHORITY}, sort_keys=True))
        return 2
    print(json.dumps({"ok": True, "result": result}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
