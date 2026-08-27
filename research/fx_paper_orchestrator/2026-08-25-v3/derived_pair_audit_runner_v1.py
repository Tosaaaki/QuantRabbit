"""Reconstruct pair-level evidence from immutable V25--V41 official seals.

The historical result files are portfolio aggregates after V27.  This runner
therefore replays only the already-sealed signal/action contracts against the
same completed BID/ASK corpus and frozen cost/inventory implementations.  It
writes new derived evidence; it never mutates or reclassifies a historical
result and it cannot admit a strategy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


MODULE_DIR = Path(__file__).resolve().parent
COMPAT_DIR = MODULE_DIR / "paper_replay_compat"
for import_root in (MODULE_DIR, COMPAT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_causal_min_spread_representative_v26 as frozen_v26  # noqa: E402
import run_causal_basket_hold_v28 as frozen_v28  # noqa: E402
import run_causal_basket_consensus_release_v29 as frozen_v29  # noqa: E402
import run_causal_consensus_release_scope_v30 as frozen_v30  # noqa: E402
import run_causal_consensus_release_persistence_v31 as frozen_v31  # noqa: E402


REGISTRY_PATH = "PAPER_RESEARCH_CYCLE_REGISTRY_V2.json"
OUTPUT_ROOT = "evidence/derived_pair_audit_v1"
METRICS_PATH = f"{OUTPUT_ROOT}/pair_month_arm_metrics_v1.jsonl"
DAILY_RANGE_PATH = f"{OUTPUT_ROOT}/completed_daily_range_oracle_inputs_v1.jsonl"
AUDIT_PATH = f"{OUTPUT_ROOT}/derived_pair_audit_v1.json"
SOURCE_ROOT = Path("/Users/tossaki/App/QuantRabbit/logs/replay/oanda_history/20260715T115624Z")
VALID_CYCLES = (
    "V25", "V27", "V28", "V29", "V30", "V31", "V33", "V35", "V37",
    "V38", "V39", "V40", "V41",
)
FAILED_CYCLE_ARTIFACTS = {
    "V26": "V26_AUTHORIZED_RECOVERY_FAILURE.json",
    "V32": "V32_OFFICIAL_EXECUTION_FAILURE.json",
    "V34": "V34_RESULT_VALIDATION_FAILURE.json",
    "V36": "V36_OFFICIAL_EXECUTION_FAILURE.json",
}
ARMS = ("RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS")
UNIVERSE = tuple(sorted(frozen_v28.UNIVERSE))
SELECTED_PAIRS = ("AUD_USD", "EUR_USD", "USD_JPY")
PERIODS = {
    "MONTH_2026_05": ("2026-05-01", "2026-06-01"),
    "MONTH_2026_06": ("2026-06-01", "2026-07-01"),
    "WALK_FORWARD": ("2026-05-01", "2026-07-01"),
}
INITIAL_EQUITY_JPY = 200_000.0
PAIR_SLEEVE = 1.0 / 7.0
MIN_COMPLETE_BARS_PER_UTC_DAY = 250
AUTHORITY = dict(frozen_v28.AUTHORITY)
EXPECTED_DEDUPE = {
    "AUD_USD": {"ledger_rows": 534, "unique_signal_ids": 167},
    "EUR_USD": {"ledger_rows": 596, "unique_signal_ids": 183},
    "USD_JPY": {"ledger_rows": 535, "unique_signal_ids": 168},
}
EXPECTED_V25_WALK_FORWARD = {
    "AUD_USD": {
        "episodes": 30,
        "direction_accuracy": 0.5,
        "jpy": {"RAW_SIGNAL": -88.30, "EXECUTABLE_BASE": -324.52, "ADVERSE_STRESS": -521.89},
    },
    "EUR_USD": {
        "episodes": 35,
        "direction_accuracy": 19 / 35,
        "jpy": {"RAW_SIGNAL": 322.79, "EXECUTABLE_BASE": 117.93, "ADVERSE_STRESS": -50.42},
    },
    "USD_JPY": {
        "episodes": 29,
        "direction_accuracy": 16 / 29,
        "jpy": {"RAW_SIGNAL": 59.67, "EXECUTABLE_BASE": -63.74, "ADVERSE_STRESS": -179.28},
    },
}
EXPECTED_RANGE = {
    "2026-05": {
        "EUR_USD": (21, 54.04523809523804, 56.59999999999776),
        "USD_JPY": (21, 79.09761904761851, 55.35000000000139),
    },
    "2026-06": {
        "EUR_USD": (22, 61.91818181818182, 54.7249999999988),
        "USD_JPY": (22, 55.21363636363646, 46.47499999999951),
    },
}
EXPECTED_ORACLE_CAPTURE_PERCENT = {
    "2026-05": {
        "1.0": 701.8913439271561, "4.0": 178.07852034679618,
        "8.0": 90.77661327318212, "12.0": 61.67615287452617,
        "20.0": 38.39609984687502,
    },
    "2026-06": {
        "1.0": 731.3337945460546, "4.0": 185.596723573916,
        "8.0": 94.64067519050916, "12.0": 64.32209150015993,
        "20.0": 40.06740285995378,
    },
}


class DerivedPairAuditError(RuntimeError):
    """A fail-closed historical evidence or reconstruction violation."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def embedded_hash(payload: dict[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(canonical_bytes(unsigned)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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


def registry_cycle_specs(root: Path) -> dict[str, dict[str, Any]]:
    registry = load_json(root / REGISTRY_PATH)
    specs = {item["cycle_id"]: item for item in registry["cycles"]}
    if tuple(cycle for cycle in VALID_CYCLES if cycle in specs) != VALID_CYCLES:
        raise DerivedPairAuditError("valid cycle registry membership changed")
    return specs


def authority_is_zero(value: dict[str, Any]) -> bool:
    return value == AUTHORITY


def verify_sealed_cycle(
    root: Path, spec: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    cycle = spec["cycle_id"]
    execution = spec["execution"]
    result_path = root / execution["result"]
    ledger_path = root / execution["ledger"]
    seal_path = root / "evidence/orchestrator_state_v2" / f"official_seal_{cycle.lower()}.json"
    if not all(path.is_file() for path in (result_path, ledger_path, seal_path)):
        raise DerivedPairAuditError(f"missing sealed artifact: {cycle}")
    result, rows, seal = load_json(result_path), load_jsonl(ledger_path), load_json(seal_path)
    if seal.get("cycle_id") != cycle or result.get("cycle_id", cycle) != cycle:
        raise DerivedPairAuditError(f"cycle identity mismatch: {cycle}")
    if seal.get("official_seal_sha256") != embedded_hash(seal, "official_seal_sha256"):
        raise DerivedPairAuditError(f"official seal self-hash mismatch: {cycle}")
    if result.get("result_sha256") != embedded_hash(result, "result_sha256"):
        raise DerivedPairAuditError(f"embedded result hash mismatch: {cycle}")
    if seal.get("embedded_result_sha256") != result["result_sha256"]:
        raise DerivedPairAuditError(f"sealed embedded result hash mismatch: {cycle}")
    if seal.get("result_file_sha256") != sha256_file(result_path):
        raise DerivedPairAuditError(f"sealed result file hash mismatch: {cycle}")
    ledger_hash = sha256_file(ledger_path)
    if seal.get("ledger_sha256") != ledger_hash \
            or result.get("proposal_ledger_sha256") != ledger_hash:
        raise DerivedPairAuditError(f"sealed ledger hash mismatch: {cycle}")
    signal_hash = frozen_v26.signal_id_set_hash(rows)
    if signal_hash != seal.get("signal_id_set_sha256"):
        raise DerivedPairAuditError(f"signal-id-set hash mismatch: {cycle}")
    if len(rows) != len({row["signal_id"] for row in rows}):
        raise DerivedPairAuditError(f"duplicate signal id inside sealed ledger: {cycle}")
    if not authority_is_zero(seal.get("authority", {})):
        raise DerivedPairAuditError(f"authority boundary changed: {cycle}")
    system = seal.get("system_acceptance", {})
    if system.get("holdout_state") != "UNOPENED" or system.get("external_orders") != 0 \
            or system.get("paper_only") is not True:
        raise DerivedPairAuditError(f"system boundary changed: {cycle}")
    gate = seal.get("strategy_profit_gate", {})
    if gate.get("passed") is not False or gate.get("adoption_authorized") is not False:
        raise DerivedPairAuditError(f"historical strategy classification changed: {cycle}")
    if result.get("external_orders") != 0 or result.get("live_authority") is not False:
        raise DerivedPairAuditError(f"result external boundary changed: {cycle}")
    result_holdout = result.get("holdout")
    if result_holdout is not None and result_holdout.get("state") != "UNOPENED":
        raise DerivedPairAuditError(f"result holdout boundary changed: {cycle}")
    for arm in ARMS:
        walk = result["periods"]["WALK_FORWARD"][arm]
        if walk.get("terminal_open_inventory") != 0:
            raise DerivedPairAuditError(f"terminal inventory changed: {cycle}/{arm}")
        if "terminal_inventory_mtm" in walk and walk["terminal_inventory_mtm"] != 0.0:
            raise DerivedPairAuditError(f"terminal MTM changed: {cycle}/{arm}")
        if "terminal_inventory_mtm" not in walk \
                and result.get("terminal_inventory_mtm_hidden") is not False:
            raise DerivedPairAuditError(f"legacy terminal MTM evidence missing: {cycle}/{arm}")
    for registry_field, seal_field in (
        ("preregistration", "preregistration_sha256"),
        ("script", "script_sha256"),
        ("test", "test_sha256"),
    ):
        source_path = root / spec[registry_field]
        expected = seal.get(seal_field)
        if not source_path.is_file() or sha256_file(source_path) != expected \
                or spec[f"{registry_field}_sha256"] != expected:
            raise DerivedPairAuditError(f"sealed source hash mismatch: {cycle}/{registry_field}")
    proof = {
        "cycle_id": cycle,
        "result_path": execution["result"],
        "result_file_sha256": sha256_file(result_path),
        "embedded_result_sha256": result["result_sha256"],
        "ledger_path": execution["ledger"],
        "ledger_sha256": ledger_hash,
        "official_seal_path": str(seal_path.relative_to(root)),
        "official_seal_file_sha256": sha256_file(seal_path),
        "official_seal_sha256": seal["official_seal_sha256"],
        "signal_id_set_sha256": signal_hash,
        "signals": len(rows),
        "source_manifest_sha256": seal["source_manifest_sha256"],
        "holdout": "UNOPENED",
        "external_orders": 0,
        "metrics_admissible": True,
    }
    return result, rows, seal, proof


def verify_failed_cycles(root: Path, specs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for cycle, relative in FAILED_CYCLE_ARTIFACTS.items():
        artifact_path = root / relative
        if not artifact_path.is_file():
            raise DerivedPairAuditError(f"missing failure evidence: {cycle}")
        artifact = load_json(artifact_path)
        profit_proven = artifact.get("profit_proven")
        if profit_proven is None:
            profit_proven = artifact.get("strategy_evidence", {}).get("profit_proven")
        if profit_proven is None:
            profit_proven = artifact.get("result_state", {}).get("profit_proven")
        if artifact.get("cycle_id") != cycle or profit_proven is not False:
            raise DerivedPairAuditError(f"invalid failure evidence: {cycle}")
        execution = specs[cycle]["execution"]
        result_path, ledger_path = root / execution["result"], root / execution["ledger"]
        seal_path = root / "evidence/orchestrator_state_v2" / f"official_seal_{cycle.lower()}.json"
        if cycle == "V34":
            if not result_path.is_file() or not ledger_path.is_file() or seal_path.exists():
                raise DerivedPairAuditError("V34 invalid-result artifact state changed")
            if artifact.get("metrics_admissible") is not False \
                    or artifact.get("official_seal_exists") is not False:
                raise DerivedPairAuditError("V34 became admissible")
            if sha256_file(result_path) != artifact["result_file_sha256"] \
                    or sha256_file(ledger_path) != artifact["ledger_file_sha256"]:
                raise DerivedPairAuditError("V34 invalid result hash changed")
        else:
            if result_path.exists() or ledger_path.exists() or seal_path.exists():
                raise DerivedPairAuditError(f"failed no-result cycle unexpectedly has output: {cycle}")
        authority = artifact.get("authority")
        if authority is not None and not authority_is_zero(authority):
            raise DerivedPairAuditError(f"failed-cycle authority changed: {cycle}")
        records.append({
            "cycle_id": cycle,
            "failure_artifact": relative,
            "failure_artifact_sha256": sha256_file(artifact_path),
            "status": artifact["status"],
            "metrics_admissible": False,
            "official_seal_exists": False,
            "excluded_from_pair_metrics": True,
            "rerun_permitted": False,
        })
    return records


def verify_and_load_corpus(
    results: dict[str, dict[str, Any]], specs: dict[str, dict[str, Any]]
) -> tuple[dict[str, list], list[dict[str, Any]]]:
    frozen_v28.runtime_v27.install_timestamp_compatibility()
    corpus, actual = frozen_v26.load_corpus(SOURCE_ROOT)
    actual_by_pair = {item["pair"]: item for item in actual}
    reference_files = specs["V25"]["source_contract"]["files"]
    reference_manifest = specs["V25"]["source_contract"]["manifest_sha256"]
    records = []
    for pair in UNIVERSE:
        item = actual_by_pair[pair]
        if item["source_sha256"] != reference_files[pair]:
            raise DerivedPairAuditError(f"actual source hash mismatch: {pair}")
        for cycle in VALID_CYCLES:
            audit = {entry["pair"]: entry for entry in results[cycle]["source_audit"]}
            if audit[pair]["source_sha256"] != item["source_sha256"] \
                    or audit[pair]["bars"] != item["bars"]:
                raise DerivedPairAuditError(f"source audit differs: {cycle}/{pair}")
            if specs[cycle]["source_contract"]["files"][pair] != item["source_sha256"] \
                    or specs[cycle]["source_contract"]["manifest_sha256"] != reference_manifest:
                raise DerivedPairAuditError(f"registry source contract differs: {cycle}/{pair}")
        records.append({
            **item,
            "source_root": str(SOURCE_ROOT),
            "completed_only": True,
            "price_component": "BID_ASK",
            "direct_hash_readback_passed": True,
        })
    return corpus, records


def direct_period_rows(rows: list[dict[str, Any]], start: str, end: str) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if start <= row["fill_time"][:10] < end and row["exit_time"][:10] < end
    ]


def build_direct_pair_plan(
    pair: str, bars: list, selected_rows: list[dict[str, Any]], start: str, end: str
) -> dict[str, Any]:
    """Mirror the frozen V15 nominal-exit episode netting with explicit episodes."""
    signals = sorted(
        [row for row in direct_period_rows(selected_rows, start, end) if row["pair"] == pair],
        key=lambda row: (row["fill_time"], row["signal_id"]),
    )
    if len({row["fill_time"] for row in signals}) != len(signals):
        raise DerivedPairAuditError(f"direct plan has duplicate fill: {pair}")
    by_fill = {row["fill_time"]: row for row in signals}
    period_bars = [bar for bar in bars if start <= bar.time[:10] < end]
    if not period_bars:
        raise DerivedPairAuditError(f"direct plan missing period bars: {pair}")
    position: dict[str, Any] | None = None
    episodes: list[dict[str, Any]] = []
    signal_events: list[dict[str, Any]] = []
    close_events: list[dict[str, Any]] = []

    def open_position(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "entry_time": row["fill_time"],
            "exit_time": row["exit_time"],
            "direction": int(row["direction"]),
            "source_signal_ids": [row["signal_id"]],
        }

    def close_position(stamp: str, exit_at_open: bool, reason: str) -> None:
        nonlocal position
        if position is None:
            raise DerivedPairAuditError("direct plan attempted to close absent inventory")
        age = frozen_v28.elapsed_seconds(position["entry_time"], stamp)
        episode = {
            "pair": pair,
            "entry_time": position["entry_time"],
            "exit_time": stamp,
            "direction": position["direction"],
            "exit_at_open": exit_at_open,
            "close_reason": reason,
            "inventory_age_seconds": age,
            "source_signal_ids": list(position["source_signal_ids"]),
        }
        episodes.append(episode)
        close_events.append({
            "event_type": reason,
            "pair": pair,
            "time": stamp,
            "entry_time": position["entry_time"],
            "direction": position["direction"],
            "exit_at_open": exit_at_open,
        })
        position = None

    for bar in period_bars:
        signal = by_fill.get(bar.time)
        if signal is not None:
            direction = int(signal["direction"])
            if position is None:
                action = "OPEN_FIXED_ONE_SEVENTH"
                position = open_position(signal)
            elif position["direction"] == direction:
                action = "IGNORE_SAME_DIRECTION_KEEP_ORIGINAL_EXIT"
                position["source_signal_ids"].append(signal["signal_id"])
            else:
                close_position(bar.time, True, "OPPOSITE_SIGNAL_CLOSE")
                position = open_position(signal)
                action = "REVERSE_FIXED_ONE_SEVENTH"
            signal_events.append({
                "signal_id": signal["signal_id"],
                "pair": pair,
                "time": bar.time,
                "direction": direction,
                "action": action,
            })
        if position is not None and position["exit_time"] == bar.time:
            close_position(bar.time, False, "NOMINAL_EXIT_CLOSE")
    if len(signal_events) != len(signals):
        raise DerivedPairAuditError(f"direct plan signal/bar mismatch: {pair}")
    if position is not None:
        close_position(period_bars[-1].time, False, "TERMINAL_LIQUIDATION")
    material = {"signal_events": signal_events, "close_events": close_events}
    return {
        "pair": pair,
        "signals": signals,
        "period_bars": period_bars,
        "signal_events": signal_events,
        "close_events": close_events,
        "episodes": episodes,
        "transition_sha256": hashlib.sha256(canonical_bytes(material)).hexdigest(),
    }


def execution_rows(cycle: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if cycle in {"V27", "V35"}:
        return [row for row in rows if row.get("execution_selected") is True]
    return rows


def build_cycle_plans(
    cycle: str, result: dict[str, Any], rows: list[dict[str, Any]],
    corpus: dict[str, list], start: str, end: str,
) -> dict[str, dict[str, Any]]:
    selected = execution_rows(cycle, rows)
    if cycle in {"V25", "V27"}:
        return {
            pair: build_direct_pair_plan(pair, corpus[pair], selected, start, end)
            for pair in UNIVERSE
        }
    builders: dict[str, Callable[..., dict[str, dict[str, Any]]]] = {
        "V28": frozen_v28.build_period_plans,
        "V29": frozen_v29.build_period_plans,
        "V30": frozen_v30.build_period_plans,
        "V31": frozen_v31.build_period_plans,
    }
    if cycle in builders:
        return builders[cycle](corpus, selected, start, end)
    target_hold = result["execution_rule"]["target_hold_seconds"]
    prior = frozen_v31.TARGET_HOLD_SECONDS
    frozen_v31.TARGET_HOLD_SECONDS = target_hold
    try:
        return frozen_v31.build_period_plans(corpus, selected, start, end)
    finally:
        frozen_v31.TARGET_HOLD_SECONDS = prior


def max_drawdown(values: list[float]) -> float:
    peak = values[0]
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1.0)
    return drawdown


def session_chronology(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def clocks(field: str) -> list[str]:
        return sorted({row[field][11:19] for row in rows})

    return {
        "decision_utc_clocks": clocks("decision_time"),
        "fill_utc_clocks": clocks("fill_time"),
        "nominal_ledger_exit_utc_clocks": clocks("exit_time"),
        "chronology": "COMPLETED_DATA_ONLY_AS_SEALED",
    }


def pair_metric_row(
    cycle: str, result: dict[str, Any], spec: dict[str, Any],
    all_rows: list[dict[str, Any]], plan: dict[str, Any],
    period: str, start: str, end: str,
) -> dict[str, Any]:
    pair = plan["pair"]
    if cycle in {"V25", "V27"}:
        raw_source = [
            row for row in direct_period_rows(all_rows, start, end) if row["pair"] == pair
        ]
    else:
        raw_source = [
            row for row in all_rows if start <= row["fill_time"][:10] < end
            and row["pair"] == pair
        ]
    marks: dict[str, dict[str, float]] = {}
    returns: dict[str, list[float]] = {}
    for arm in ARMS:
        marks[arm], _active, _direction, returns[arm] = frozen_v28._pair_marks(plan, arm)
    if len({len(returns[arm]) for arm in ARMS}) != 1:
        raise DerivedPairAuditError(f"cost-arm episode mismatch: {cycle}/{period}/{pair}")
    raw_returns = returns["RAW_SIGNAL"]
    arm_payload: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        net_returns = returns[arm]
        values = [marks[arm][stamp] for stamp in sorted(marks[arm], key=frozen_v31.ns)]
        multiple = values[-1]
        gross_bps = statistics.fmean(raw_returns) * 10_000.0 if raw_returns else None
        net_bps = statistics.fmean(net_returns) * 10_000.0 if net_returns else None
        cost_bps = statistics.fmean(
            gross - net for gross, net in zip(raw_returns, net_returns)
        ) * 10_000.0 if net_returns else None
        jpy = INITIAL_EQUITY_JPY * PAIR_SLEEVE * (multiple - 1.0)
        raw_multiple = [marks["RAW_SIGNAL"][stamp]
                        for stamp in sorted(marks["RAW_SIGNAL"], key=frozen_v31.ns)][-1]
        arm_payload[arm] = {
            "pair_standalone_equity_multiple": multiple,
            "fixed_one_seventh_capital_contribution_multiple": 1.0 + (multiple - 1.0) * PAIR_SLEEVE,
            "ending_sleeve_equity_jpy": INITIAL_EQUITY_JPY * PAIR_SLEEVE * multiple,
            "jpy_contribution_legacy_sealed_convention": jpy,
            "gross_edge_bps_per_realized_episode": gross_bps,
            "net_edge_bps_per_realized_episode": net_bps,
            "realized_cost_drag_bps_per_realized_episode": cost_bps,
            "compounded_cost_drag_jpy_from_raw": (
                INITIAL_EQUITY_JPY * PAIR_SLEEVE * (raw_multiple - multiple)
            ),
            "cost_to_expected_move_ratio": (
                cost_bps / gross_bps if gross_bps is not None and gross_bps > 0 else None
            ),
            "max_drawdown": max_drawdown(values),
            "terminal_open_inventory": 0,
            "terminal_inventory_mtm": 0.0,
        }
    action_counts = Counter(event["action"] for event in plan["signal_events"])
    family = spec["hypothesis_contract"]["family"]
    executed_days = len({row["utc_day"] for row in plan["signals"]})
    source_days = len({row["utc_day"] for row in raw_source})
    return {
        "cycle_id": cycle,
        "period": period,
        "start": start,
        "end_exclusive": end,
        "pair": pair,
        "family": family,
        "source_timeframe": spec["source_contract"]["bar_granularity"],
        "source_price_component": spec["source_contract"]["price_component"],
        "session_chronology": session_chronology(raw_source) if raw_source else {
            "decision_utc_clocks": [], "fill_utc_clocks": [],
            "nominal_ledger_exit_utc_clocks": [],
            "chronology": "NO_PAIR_SIGNAL_IN_PERIOD",
        },
        "raw_signals": len(raw_source),
        "selected_signals": len(plan["signals"]),
        "executed_episodes": len(plan["episodes"]),
        "direction_accuracy": (
            sum(value > 0 for value in raw_returns) / len(raw_returns) if raw_returns else None
        ),
        "N_eff": {
            "observed_raw_signal_days": source_days,
            "observed_executed_signal_days": executed_days,
            "observed_nonoverlapping_pair_episodes": len(plan["episodes"]),
            "autocorrelation_adjusted": None,
            "common_currency_time_cluster_adjusted": None,
            "classification": "OBSERVED_COUNTS_ONLY_NOT_AN_INDEPENDENCE_CLAIM",
        },
        "turnover_nav_at_fixed_one_seventh": 2.0 * len(plan["episodes"]) * PAIR_SLEEVE,
        "max_inventory_age_seconds": max(
            [0.0] + [episode["inventory_age_seconds"] for episode in plan["episodes"]]
        ),
        "action_counts": dict(sorted(action_counts.items())),
        "execution_transition_sha256": plan["transition_sha256"],
        "accounting_classification": "DERIVED_USING_LEGACY_SEALED_RETURN_CONVENTION_PENDING_JPY_MIGRATION",
        "sealed_result_reclassified": False,
        "arms": arm_payload,
    }


def reconstruct_all_pairs(
    specs: dict[str, dict[str, Any]], results: dict[str, dict[str, Any]],
    ledgers: dict[str, list[dict[str, Any]]], corpus: dict[str, list],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    reconciliation: dict[str, Any] = {}
    for cycle in VALID_CYCLES:
        result, rows, spec = results[cycle], ledgers[cycle], specs[cycle]
        for period, (start, end) in PERIODS.items():
            plans = build_cycle_plans(cycle, result, rows, corpus, start, end)
            if set(plans) != set(UNIVERSE):
                raise DerivedPairAuditError(f"planner universe changed: {cycle}/{period}")
            period_rows = [
                pair_metric_row(cycle, result, spec, rows, plans[pair], period, start, end)
                for pair in UNIVERSE
            ]
            metrics.extend(period_rows)
            arm_reconciliation = {}
            for arm in ARMS:
                reconstructed = statistics.fmean(
                    row["arms"][arm]["pair_standalone_equity_multiple"]
                    for row in period_rows
                )
                sealed = result["periods"][period][arm]["equity_multiple"]
                difference = abs(reconstructed - sealed)
                if not math.isclose(reconstructed, sealed, rel_tol=0.0, abs_tol=1e-12):
                    raise DerivedPairAuditError(
                        f"pair reconstruction mismatch: {cycle}/{period}/{arm}: "
                        f"{reconstructed} != {sealed}"
                    )
                arm_reconciliation[arm] = {
                    "reconstructed_portfolio_equity_multiple": reconstructed,
                    "sealed_portfolio_equity_multiple": sealed,
                    "absolute_difference": difference,
                    "passed": True,
                }
                if cycle in {"V25", "V27"}:
                    stored = result["periods"][period][arm]["pair_audit"]
                    for row in period_rows:
                        pair = row["pair"]
                        reconstructed_pair = row["arms"][arm]["pair_standalone_equity_multiple"]
                        if not math.isclose(
                            reconstructed_pair, stored[pair]["sleeve_equity_multiple"],
                            rel_tol=0.0, abs_tol=1e-12,
                        ):
                            raise DerivedPairAuditError(
                                f"stored pair audit mismatch: {cycle}/{period}/{arm}/{pair}"
                            )
            reconciliation[f"{cycle}:{period}"] = arm_reconciliation
    return metrics, reconciliation


def verify_v25_pair_evidence(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {
        row["pair"]: row for row in metrics
        if row["cycle_id"] == "V25" and row["period"] == "WALK_FORWARD"
        and row["pair"] in SELECTED_PAIRS
    }
    records = {}
    for pair, expected in EXPECTED_V25_WALK_FORWARD.items():
        row = indexed[pair]
        if row["executed_episodes"] != expected["episodes"] \
                or not math.isclose(row["direction_accuracy"], expected["direction_accuracy"],
                                    rel_tol=0.0, abs_tol=1e-15):
            raise DerivedPairAuditError(f"V25 pair count/accuracy changed: {pair}")
        jpy = {}
        for arm, rounded_expected in expected["jpy"].items():
            value = row["arms"][arm]["jpy_contribution_legacy_sealed_convention"]
            if not math.isclose(value, rounded_expected, rel_tol=0.0, abs_tol=0.0051):
                raise DerivedPairAuditError(
                    f"V25 pair JPY contribution changed: {pair}/{arm}: {value}"
                )
            jpy[arm] = value
        records[pair] = {
            "executed_episodes": row["executed_episodes"],
            "direction_accuracy": row["direction_accuracy"],
            "jpy_contribution_legacy_sealed_convention": jpy,
            "matches_independent_direct_readback": True,
        }
    v27 = {
        row["pair"]: row for row in metrics
        if row["cycle_id"] == "V27" and row["period"] == "WALK_FORWARD"
        and row["pair"] in SELECTED_PAIRS
    }
    if v27["EUR_USD"]["executed_episodes"] != 1 \
            or v27["AUD_USD"]["executed_episodes"] != 0 \
            or v27["USD_JPY"]["executed_episodes"] != EXPECTED_V25_WALK_FORWARD["USD_JPY"]["episodes"]:
        raise DerivedPairAuditError("V27 selected pair episode readback changed")
    for arm in ("EXECUTABLE_BASE", "ADVERSE_STRESS"):
        if v27["EUR_USD"]["arms"][arm]["jpy_contribution_legacy_sealed_convention"] <= 0:
            raise DerivedPairAuditError(f"V27 EUR single episode is no longer positive: {arm}")
    return {
        "V25": records,
        "V27": {
            pair: {
                "executed_episodes": row["executed_episodes"],
                "base_jpy": row["arms"]["EXECUTABLE_BASE"][
                    "jpy_contribution_legacy_sealed_convention"
                ],
                "adverse_jpy": row["arms"]["ADVERSE_STRESS"][
                    "jpy_contribution_legacy_sealed_convention"
                ],
            }
            for pair, row in sorted(v27.items())
        },
    }


def dedupe_audit(
    seals: dict[str, dict[str, Any]], ledgers: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    pair_ids: dict[str, list[str]] = {pair: [] for pair in SELECTED_PAIRS}
    for cycle in VALID_CYCLES:
        for row in ledgers[cycle]:
            if row["pair"] in pair_ids:
                pair_ids[row["pair"]].append(row["signal_id"])
    pair_counts = {
        pair: {"ledger_rows": len(ids), "unique_signal_ids": len(set(ids))}
        for pair, ids in sorted(pair_ids.items())
    }
    if pair_counts != EXPECTED_DEDUPE:
        raise DerivedPairAuditError(f"selected-pair dedupe counts changed: {pair_counts}")
    groups: dict[str, list[str]] = defaultdict(list)
    for cycle in VALID_CYCLES:
        groups[seals[cycle]["signal_id_set_sha256"]].append(cycle)
    if len(groups) != 6:
        raise DerivedPairAuditError(f"independent RAW stream count changed: {len(groups)}")
    return {
        "valid_sealed_cycles": list(VALID_CYCLES),
        "valid_sealed_cycle_count": len(VALID_CYCLES),
        "pair_signal_id_dedupe": pair_counts,
        "unique_signal_id_set_count": len(groups),
        "signal_id_set_groups": [
            {"signal_id_set_sha256": key, "cycles": value, "cycle_count": len(value)}
            for key, value in sorted(groups.items(), key=lambda item: item[1][0])
        ],
        "cycle_count_is_independent_trial_count": False,
        "classification": "SIX_DISTINCT_RAW_STREAMS_THIRTEEN_VALID_SEALED_CYCLES",
    }


def daily_range_and_oracle(corpus: dict[str, list]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for pair in ("EUR_USD", "USD_JPY"):
        for bar in corpus[pair]:
            month = bar.time[:7]
            if month in EXPECTED_RANGE:
                grouped[month][pair][bar.time[:10]].append(bar)
    rows: list[dict[str, Any]] = []
    eligible: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    summaries: dict[str, Any] = {}
    for month in sorted(EXPECTED_RANGE):
        summaries[month] = {"pairs": {}}
        for pair in ("EUR_USD", "USD_JPY"):
            pip = 0.01 if pair.endswith("_JPY") else 0.0001
            for utc_day, bars in sorted(grouped[month][pair].items()):
                if len(bars) < MIN_COMPLETE_BARS_PER_UTC_DAY:
                    continue
                high = max(bar.mid_h for bar in bars)
                low = min(bar.mid_l for bar in bars)
                midrange = (high + low) / 2.0
                range_pips = (high - low) / pip
                log_range = math.log(high / low)
                median_spread_pips = statistics.median(
                    (bar.ask_c - bar.bid_c) / pip for bar in bars
                )
                cost_fraction = (median_spread_pips + 0.6) * pip / midrange
                item = {
                    "month": month,
                    "utc_day": utc_day,
                    "pair": pair,
                    "completed_m5_bid_ask_bars": len(bars),
                    "eligible_minimum_250_bars": True,
                    "midpoint_high": high,
                    "midpoint_low": low,
                    "midrange": midrange,
                    "daily_range_pips": range_pips,
                    "daily_log_range": log_range,
                    "median_completed_close_spread_pips": median_spread_pips,
                    "slippage_pips_per_side": 0.3,
                    "roundtrip_cost_fraction": cost_fraction,
                    "range_known_at_entry": False,
                }
                rows.append(item)
                eligible[month][pair][utc_day] = item
            pair_rows = [row for row in rows if row["month"] == month and row["pair"] == pair]
            values = [row["daily_range_pips"] for row in pair_rows]
            expected_days, expected_mean, expected_median = EXPECTED_RANGE[month][pair]
            summary = {
                "eligible_utc_days": len(values),
                "mean_daily_range_pips": statistics.fmean(values),
                "median_daily_range_pips": statistics.median(values),
            }
            if summary["eligible_utc_days"] != expected_days \
                    or not math.isclose(summary["mean_daily_range_pips"], expected_mean,
                                        rel_tol=0.0, abs_tol=1e-12) \
                    or not math.isclose(summary["median_daily_range_pips"], expected_median,
                                        rel_tol=0.0, abs_tol=1e-12):
                raise DerivedPairAuditError(f"daily range readback changed: {month}/{pair}")
            summaries[month]["pairs"][pair] = summary
        common_days = sorted(set(eligible[month]["EUR_USD"]) & set(eligible[month]["USD_JPY"]))
        if len(common_days) != EXPECTED_RANGE[month]["EUR_USD"][0]:
            raise DerivedPairAuditError(f"oracle common-day set changed: {month}")

        def monthly_multiple(capture_fraction: float, gross_cap: float) -> float:
            wealth = 1.0
            for day in common_days:
                pair_sum = sum(
                    capture_fraction * eligible[month][pair][day]["daily_log_range"]
                    - eligible[month][pair][day]["roundtrip_cost_fraction"]
                    for pair in ("EUR_USD", "USD_JPY")
                )
                daily_return = gross_cap / 2.0 * pair_sum
                if daily_return <= -1.0:
                    return 0.0
                wealth *= 1.0 + daily_return
            return wealth

        solutions = {}
        for gross_cap in (1.0, 4.0, 8.0, 12.0, 20.0):
            lower, upper = 0.0, 1.0
            while monthly_multiple(upper, gross_cap) < 2.0:
                upper *= 2.0
                if upper > 1024.0:
                    raise DerivedPairAuditError("oracle capture bisection did not bracket 2x")
            for _ in range(200):
                midpoint = (lower + upper) / 2.0
                if monthly_multiple(midpoint, gross_cap) < 2.0:
                    lower = midpoint
                else:
                    upper = midpoint
            capture = (lower + upper) / 2.0
            capture_percent = capture * 100.0
            expected = EXPECTED_ORACLE_CAPTURE_PERCENT[month][str(gross_cap)]
            if not math.isclose(capture_percent, expected, rel_tol=0.0, abs_tol=1e-10):
                raise DerivedPairAuditError(
                    f"oracle capture readback changed: {month}/{gross_cap}: {capture_percent}"
                )
            solutions[str(gross_cap)] = {
                "gross_cap": gross_cap,
                "required_daily_high_low_capture_fraction": capture,
                "required_daily_high_low_capture_percent": capture_percent,
                "solved_monthly_multiple": monthly_multiple(capture, gross_cap),
                "gross_cap_authorized": gross_cap == 1.0,
                "strategy_evidence": False,
                "perfect_full_range_normal_cost_ceiling_multiple": monthly_multiple(1.0, gross_cap),
            }
        summaries[month]["common_eligible_utc_days"] = len(common_days)
        summaries[month]["oracle_capture_solutions"] = solutions
    return rows, {
        "classification": "LOOKAHEAD_UPPER_BOUND_DIAGNOSTIC_NOT_STRATEGY_EVIDENCE",
        "completed_daily_range": summaries,
        "current_authorized_gross_cap": 1.0,
        "current_1x_required_capture_exceeds_full_daily_range": all(
            summaries[month]["oracle_capture_solutions"]["1.0"]
            ["required_daily_high_low_capture_fraction"] > 1.0
            for month in summaries
        ),
        "gross_cap_change_authorized": False,
        "evaluated_gross_caps": [1.0, 4.0, 8.0, 12.0, 20.0],
        "full_daily_high_low_is_lookahead": True,
        "every_day_positive_capture_is_unrealistic": True,
        "may_admit_strategy": False,
    }


def protected_hashes(root: Path, specs: dict[str, dict[str, Any]]) -> dict[str, str]:
    paths = []
    for cycle in VALID_CYCLES:
        paths.extend((
            specs[cycle]["execution"]["result"],
            specs[cycle]["execution"]["ledger"],
            f"evidence/orchestrator_state_v2/official_seal_{cycle.lower()}.json",
        ))
    paths.extend(FAILED_CYCLE_ARTIFACTS.values())
    paths.append("evidence/orchestrator_state_v2/next_hypothesis_work_order_v42.json")
    return {relative: sha256_file(root / relative) for relative in sorted(paths)}


def build(root: Path) -> dict[str, Any]:
    specs = registry_cycle_specs(root)
    before = protected_hashes(root, specs)
    results: dict[str, dict[str, Any]] = {}
    ledgers: dict[str, list[dict[str, Any]]] = {}
    seals: dict[str, dict[str, Any]] = {}
    proofs = []
    for cycle in VALID_CYCLES:
        result, rows, seal, proof = verify_sealed_cycle(root, specs[cycle])
        results[cycle], ledgers[cycle], seals[cycle] = result, rows, seal
        proofs.append(proof)
    failed = verify_failed_cycles(root, specs)
    corpus, source_readback = verify_and_load_corpus(results, specs)
    metrics, reconciliation = reconstruct_all_pairs(specs, results, ledgers, corpus)
    direct_pair_checks = verify_v25_pair_evidence(metrics)
    dedupe = dedupe_audit(seals, ledgers)
    daily_rows, range_oracle = daily_range_and_oracle(corpus)

    metrics_path = root / METRICS_PATH
    daily_path = root / DAILY_RANGE_PATH
    atomic_text(metrics_path, "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in metrics
    ))
    atomic_text(daily_path, "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in daily_rows
    ))
    after = protected_hashes(root, specs)
    if before != after:
        raise DerivedPairAuditError("historical artifact changed during derived audit")
    payload = {
        "schema_version": 1,
        "audit_id": "FX_SEALED_DERIVED_PAIR_AUDIT_V1",
        "classification": "NON_STRATEGY_READ_ONLY_DERIVED_LEGACY_EVIDENCE",
        "runner_path": Path(__file__).name,
        "runner_file_sha256": sha256_file(Path(__file__)),
        "registry_path": REGISTRY_PATH,
        "registry_file_sha256": sha256_file(root / REGISTRY_PATH),
        "valid_seal_proofs": proofs,
        "failed_and_invalid_cycles": failed,
        "deduplication": dedupe,
        "source_readback": source_readback,
        "pair_metrics_path": METRICS_PATH,
        "pair_metrics_file_sha256": sha256_file(metrics_path),
        "pair_metrics_rows": len(metrics),
        "pair_metrics_dimensions": {
            "cycles": len(VALID_CYCLES), "periods": len(PERIODS),
            "pairs": len(UNIVERSE), "arms_nested_per_row": len(ARMS),
        },
        "portfolio_reconciliation": reconciliation,
        "direct_pair_checks": direct_pair_checks,
        "daily_range_inputs_path": DAILY_RANGE_PATH,
        "daily_range_inputs_file_sha256": sha256_file(daily_path),
        "daily_range_input_rows": len(daily_rows),
        "daily_range_and_oracle_feasibility": range_oracle,
        "n_eff_limit": {
            "about_three_pair_calendar_days_across_six_streams": "40_TO_42_OBSERVED_BY_USER_AUDIT",
            "autocorrelation_and_shared_usd_adjusted_n_eff_saved_in_historical_seals": False,
            "derived_runner_may_invent_adjusted_n_eff": False,
            "cycle_count_may_not_be_used_as_independent_trial_count": True,
        },
        "accounting_convention": {
            "classification": "LEGACY_SEALED_RETURN_CONVENTION",
            "fixed_pair_sleeve": PAIR_SLEEVE,
            "initial_equity_jpy": INITIAL_EQUITY_JPY,
            "quote_to_jpy_conversion_present": False,
            "short_fixed_notional_linear_pnl_present": False,
            "may_be_used_as_new_jpy_accounting_seal": False,
            "migration_required_before_next_official_strategy_run": True,
        },
        "historical_artifact_hashes": {"before": before, "after": after, "unchanged": True},
        "holdout_state": "UNOPENED",
        "strategy_adoption_authorized": False,
        "profit_gate_pass_inferred": False,
        "official_strategy_run_performed": False,
        "external_orders": 0,
        "authority": AUTHORITY,
    }
    payload["audit_sha256"] = embedded_hash(payload, "audit_sha256")
    atomic_text(root / AUDIT_PATH, json.dumps(payload, indent=2, sort_keys=True,
                                              allow_nan=False) + "\n")
    return payload


def validate(root: Path) -> dict[str, Any]:
    payload = load_json(root / AUDIT_PATH)
    if payload.get("audit_sha256") != embedded_hash(payload, "audit_sha256"):
        raise DerivedPairAuditError("derived audit embedded hash mismatch")
    runner_path = root / payload.get("runner_path", "")
    if not runner_path.is_file() or sha256_file(runner_path) != payload.get("runner_file_sha256"):
        raise DerivedPairAuditError("derived audit runner hash mismatch")
    for path_field, hash_field in (
        ("pair_metrics_path", "pair_metrics_file_sha256"),
        ("daily_range_inputs_path", "daily_range_inputs_file_sha256"),
    ):
        path = root / payload[path_field]
        if not path.is_file() or sha256_file(path) != payload[hash_field]:
            raise DerivedPairAuditError(f"derived evidence hash mismatch: {path_field}")
    if payload["deduplication"]["valid_sealed_cycle_count"] != 13 \
            or payload["deduplication"]["unique_signal_id_set_count"] != 6:
        raise DerivedPairAuditError("13-seal/6-stream contract changed")
    if payload["deduplication"]["pair_signal_id_dedupe"] != EXPECTED_DEDUPE:
        raise DerivedPairAuditError("pair dedupe validation changed")
    if len(payload["failed_and_invalid_cycles"]) != 4 \
            or any(item["metrics_admissible"] for item in payload["failed_and_invalid_cycles"]):
        raise DerivedPairAuditError("failed-cycle exclusion changed")
    if len(payload["portfolio_reconciliation"]) != len(VALID_CYCLES) * len(PERIODS):
        raise DerivedPairAuditError("portfolio reconciliation coverage changed")
    if payload["pair_metrics_rows"] != len(VALID_CYCLES) * len(PERIODS) * len(UNIVERSE):
        raise DerivedPairAuditError("pair metric row coverage changed")
    if payload["daily_range_and_oracle_feasibility"] \
            ["current_1x_required_capture_exceeds_full_daily_range"] is not True:
        raise DerivedPairAuditError("current 1x structural ceiling changed")
    if payload.get("holdout_state") != "UNOPENED" or payload.get("external_orders") != 0 \
            or not authority_is_zero(payload.get("authority", {})):
        raise DerivedPairAuditError("derived audit boundary changed")
    if payload.get("strategy_adoption_authorized") is not False \
            or payload.get("profit_gate_pass_inferred") is not False \
            or payload.get("official_strategy_run_performed") is not False:
        raise DerivedPairAuditError("derived audit overstates profit evidence")
    specs = registry_cycle_specs(root)
    current = protected_hashes(root, specs)
    protected = payload["historical_artifact_hashes"]
    if protected.get("unchanged") is not True or protected["before"] != current \
            or protected["after"] != current:
        raise DerivedPairAuditError("historical artifact readback changed")
    return payload


def main() -> int:
    root = MODULE_DIR
    payload = build(root)
    readback = validate(root)
    if payload["audit_sha256"] != readback["audit_sha256"]:
        raise DerivedPairAuditError("derived audit build/readback mismatch")
    print(json.dumps({
        "audit_path": AUDIT_PATH,
        "audit_file_sha256": sha256_file(root / AUDIT_PATH),
        "audit_sha256": payload["audit_sha256"],
        "valid_sealed_cycles": 13,
        "unique_raw_streams": 6,
        "pair_metric_rows": payload["pair_metrics_rows"],
        "holdout": payload["holdout_state"],
        "official_strategy_run_performed": False,
        "authority": AUTHORITY,
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
