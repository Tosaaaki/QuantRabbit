"""Build hash-bound evidence for the JPY-accounting and London-DST migration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import derived_pair_audit_runner_v1 as derived
import jpy_accounting_reference_v1 as reference
import jpy_accounting_v2 as accounting
import london_session_chronology_v1 as chronology


POLICY_PATH = "JPY_ACCOUNTING_DST_RUNTIME_MIGRATION_V1.json"
OUTPUT_ROOT = "evidence/jpy_accounting_dst_runtime_migration_v1"
FIXTURE_PATH = f"{OUTPUT_ROOT}/reference_fixture_parity_v1.jsonl"
DIAGNOSTIC_ROWS_PATH = f"{OUTPUT_ROOT}/sealed_plan_accounting_diagnostic_v1.jsonl"
AUDIT_PATH = f"{OUTPUT_ROOT}/jpy_accounting_dst_runtime_migration_v1.json"
WALK_FORWARD = ("2026-05-01", "2026-07-01")
DIAGNOSTIC_CYCLES = ("V38", "V40", "V41")
PAIR_NOTIONAL_JPY = derived.INITIAL_EQUITY_JPY * derived.PAIR_SLEEVE
AUTHORITY = dict(derived.AUTHORITY)
PROTECTED = {
    "evidence/run_london_open_false_break_reclaim_v41_official_001/result_london_open_false_break_reclaim_v41.json":
        "953370828c05dbbbae6fdde1287bfd81bc3ca0317fdc519f3d13da854a60a83a",
    "evidence/run_london_open_false_break_reclaim_v41_official_001/proposal_ledger_london_open_false_break_reclaim_v41.jsonl":
        "0a6e4ce7f1198969a5cffab865c3cd7e7240f233926eab18897221afe019c600",
    "evidence/orchestrator_state_v2/official_seal_v41.json":
        "ad3362849a979f13afa3c5c1f6c52b88431a1fac54a2889839c3a93c991afc6c",
    "evidence/orchestrator_state_v2/next_hypothesis_work_order_v42.json":
        "29d541646f57efffe543007d94ce0958a2fa3cc68180cf521101886a8b09b524",
}


class MigrationEvidenceError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
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


def add_seconds(value: str, seconds: int) -> str:
    parsed = accounting.parse_utc_nanoseconds(value).value + seconds * 1_000_000_000
    whole, fraction = divmod(parsed, 1_000_000_000)
    head = datetime.fromtimestamp(whole, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{head}.{fraction:09d}Z"


def _bar_bbo(bar: Any, *, at_open: bool, available_at: str) -> accounting.BBO:
    return accounting.BBO(
        bar.pair,
        available_at,
        bar.bid_o if at_open else bar.bid_c,
        bar.ask_o if at_open else bar.ask_c,
    )


def _conversion_instruments(quote_currency: str) -> tuple[str, ...]:
    if quote_currency == "JPY":
        return ()
    if quote_currency == "USD":
        return ("USD_JPY",)
    if quote_currency == "CAD":
        return ("USD_CAD", "USD_JPY")
    if quote_currency == "CHF":
        return ("USD_CHF", "USD_JPY")
    raise MigrationEvidenceError(f"unsupported sealed-universe quote currency: {quote_currency}")


def _episode_market(
    episode: dict[str, Any],
    indexes: dict[str, dict[str, Any]],
) -> tuple[accounting.BBO, accounting.BBO, accounting.ConversionBook, str]:
    pair = episode["pair"]
    entry_stamp, exit_stamp = episode["entry_time"], episode["exit_time"]
    try:
        entry_bar = indexes[pair][entry_stamp]
        exit_bar = indexes[pair][exit_stamp]
    except KeyError as error:
        raise MigrationEvidenceError(f"sealed episode bar missing: {pair}/{error}") from error
    exit_at_open = bool(episode["exit_at_open"])
    exit_available_at = exit_stamp if exit_at_open else add_seconds(exit_stamp, 300)
    entry_bbo = _bar_bbo(entry_bar, at_open=True, available_at=entry_stamp)
    exit_bbo = _bar_bbo(
        exit_bar, at_open=exit_at_open, available_at=exit_available_at
    )
    quote_currency = accounting.pair_currencies(pair)[1]
    events: list[accounting.BBO] = []
    for instrument in _conversion_instruments(quote_currency):
        try:
            conversion_entry = indexes[instrument][entry_stamp]
            conversion_exit = indexes[instrument][exit_stamp]
        except KeyError as error:
            raise MigrationEvidenceError(
                f"causal conversion bar missing: {instrument}/{error}"
            ) from error
        events.append(_bar_bbo(
            conversion_entry, at_open=True, available_at=entry_stamp
        ))
        events.append(_bar_bbo(
            conversion_exit, at_open=exit_at_open, available_at=exit_available_at
        ))
    return entry_bbo, exit_bbo, accounting.ConversionBook(events), exit_available_at


def _fixture_rows() -> list[dict[str, Any]]:
    t0 = "2026-05-29T12:00:00.000000000Z"
    t1 = "2026-05-29T18:00:00.000000000Z"
    definitions = [
        {
            "fixture": "EUR_USD_LONG",
            "pair": "EUR_USD", "direction": 1,
            "entry": (1.1000, 1.1002), "exit": (1.1010, 1.1012),
            "conversion_entry": {"USD_JPY": (150.00, 150.02)},
            "conversion_exit": {"USD_JPY": (151.00, 151.03)},
            "path": (("USD_JPY", "USD", "JPY"),),
        },
        {
            "fixture": "EUR_USD_SHORT",
            "pair": "EUR_USD", "direction": -1,
            "entry": (1.1000, 1.1002), "exit": (1.0990, 1.0992),
            "conversion_entry": {"USD_JPY": (150.00, 150.02)},
            "conversion_exit": {"USD_JPY": (149.00, 149.03)},
            "path": (("USD_JPY", "USD", "JPY"),),
        },
        {
            "fixture": "USD_JPY_DIRECT",
            "pair": "USD_JPY", "direction": 1,
            "entry": (150.00, 150.02), "exit": (151.00, 151.03),
            "conversion_entry": {}, "conversion_exit": {}, "path": (),
        },
        {
            "fixture": "USD_CAD_TWO_HOP",
            "pair": "USD_CAD", "direction": -1,
            "entry": (1.3500, 1.3504), "exit": (1.3400, 1.3404),
            "conversion_entry": {
                "USD_CAD": (1.3500, 1.3504), "USD_JPY": (150.00, 150.02),
            },
            "conversion_exit": {
                "USD_CAD": (1.3400, 1.3404), "USD_JPY": (151.00, 151.03),
            },
            "path": (
                ("USD_CAD", "CAD", "USD"), ("USD_JPY", "USD", "JPY"),
            ),
        },
    ]
    rows = []
    scenario = accounting.ADVERSE_STRESS
    for item in definitions:
        events = []
        for instrument, values in item["conversion_entry"].items():
            events.append(accounting.BBO(instrument, t0, *values))
        for instrument, values in item["conversion_exit"].items():
            events.append(accounting.BBO(instrument, t1, *values))
        book = accounting.ConversionBook(events)
        entry = accounting.BBO(item["pair"], t0, *item["entry"])
        exit_bbo = accounting.BBO(item["pair"], t1, *item["exit"])
        position = accounting.size_position(
            item["fixture"], item["pair"], item["direction"], 28_000.0,
            t0, entry, book,
        )
        production = accounting.evaluate_position(position, t1, exit_bbo, book, scenario)
        scalar = reference.episode(
            pair=item["pair"], direction=item["direction"], notional_jpy=28_000.0,
            entry_bid=item["entry"][0], entry_ask=item["entry"][1],
            exit_bid=item["exit"][0], exit_ask=item["exit"][1],
            entry_conversion_quotes=item["conversion_entry"],
            exit_conversion_quotes=item["conversion_exit"],
            quote_to_jpy_path=item["path"], entry_time=t0, exit_time=t1,
            slippage_pips=scenario.slippage_pips_per_side,
            commission_bps_per_side=scenario.commission_bps_per_side,
            financing_bps_per_day=scenario.financing_bps_per_day,
            raw_pair_mid=scenario.raw_pair_mid,
        )
        fields = (
            "units", "raw_quote_pnl", "gross_jpy", "executable_pair_quote_pnl",
            "executable_pair_jpy", "commission_cost_jpy", "financing_cost_jpy", "net_jpy",
        )
        differences = {field: production[field] - scalar[field] for field in fields}
        if any(abs(value) > 1e-9 for value in differences.values()):
            raise MigrationEvidenceError(f"reference fixture parity failed: {item['fixture']}")
        rows.append({
            "fixture": item["fixture"],
            "pair": item["pair"],
            "direction": item["direction"],
            "scenario": scenario.name,
            "production_evaluation_sha256": production["evaluation_sha256"],
            "production_values": {field: production[field] for field in fields},
            "reference_values": {field: scalar[field] for field in fields},
            "differences": differences,
            "parity": True,
            "account_currency_midpoint_conversion_used": False,
        })
    return rows


def _sealed_inputs(root: Path) -> tuple[
    dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list]
]:
    derived.validate(root)
    specs = derived.registry_cycle_specs(root)
    results: dict[str, dict[str, Any]] = {}
    ledgers: dict[str, list[dict[str, Any]]] = {}
    for cycle in derived.VALID_CYCLES:
        result, rows, _seal, _proof = derived.verify_sealed_cycle(root, specs[cycle])
        results[cycle], ledgers[cycle] = result, rows
    corpus, _source_audit = derived.verify_and_load_corpus(results, specs)
    return results, ledgers, corpus


def _diagnostic_rows_and_summary(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results, ledgers, corpus = _sealed_inputs(root)
    indexes = {pair: {bar.time: bar for bar in bars} for pair, bars in corpus.items()}
    cause = json.loads((root / derived.OUTPUT_ROOT.replace(
        "derived_pair_audit_v1", "profit_gate_cause_feasibility_v1"
    ) / "profit_gate_cause_feasibility_audit_v1.json").read_text(encoding="utf-8"))
    prior = cause["legacy_accounting_read_only_diagnostic"]["cycles"]
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for cycle in DIAGNOSTIC_CYCLES:
        result, ledger = results[cycle], ledgers[cycle]
        plans = derived.build_cycle_plans(
            cycle, result, ledger, corpus, *WALK_FORWARD
        )
        transition_hash = hashlib.sha256(canonical_bytes({
            pair: plans[pair]["transition_sha256"] for pair in sorted(plans)
        })).hexdigest()
        episode_count = sum(len(plan["episodes"]) for plan in plans.values())
        cycle_summary: dict[str, Any] = {
            "execution_transition_sha256": transition_hash,
            "episode_count": episode_count,
            "arms": {},
        }
        units_by_episode: dict[str, float] = {}
        for arm in ("RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS"):
            scenario = accounting.SCENARIOS[arm]
            net_total = gross_total = cost_total = conversion_width_total = 0.0
            for pair in sorted(plans):
                for ordinal, episode in enumerate(plans[pair]["episodes"], 1):
                    episode_id = f"{cycle}::{pair}::{ordinal:04d}"
                    entry_bbo, exit_bbo, book, exit_available_at = _episode_market(
                        episode, indexes
                    )
                    position = accounting.size_position(
                        episode_id, pair, int(episode["direction"]),
                        PAIR_NOTIONAL_JPY, episode["entry_time"], entry_bbo, book,
                    )
                    if episode_id in units_by_episode:
                        if abs(units_by_episode[episode_id] - position.units) > 1e-12:
                            raise MigrationEvidenceError("cost arms changed explicit units")
                    else:
                        units_by_episode[episode_id] = position.units
                    evaluated = accounting.evaluate_position(
                        position, exit_available_at, exit_bbo, book, scenario
                    )
                    gross_total += evaluated["gross_jpy"]
                    net_total += evaluated["net_jpy"]
                    cost_total += evaluated["total_realized_cost_jpy"]
                    conversion_width_total += evaluated["conversion_bid_ask_width_jpy"]
                    rows.append({
                        "cycle_id": cycle,
                        "episode_id": episode_id,
                        "pair": pair,
                        "direction": int(episode["direction"]),
                        "source_signal_ids": episode["source_signal_ids"],
                        "sealed_entry_time": episode["entry_time"],
                        "sealed_exit_time": episode["exit_time"],
                        "exit_at_open": bool(episode["exit_at_open"]),
                        "accounting_exit_or_mark_available_at": exit_available_at,
                        "scenario": arm,
                        "fixed_notional_jpy": PAIR_NOTIONAL_JPY,
                        "units": position.units,
                        "gross_jpy": evaluated["gross_jpy"],
                        "net_jpy": evaluated["net_jpy"],
                        "total_realized_cost_jpy": evaluated["total_realized_cost_jpy"],
                        "conversion_bid_ask_width_jpy": evaluated[
                            "conversion_bid_ask_width_jpy"
                        ],
                        "evaluation_sha256": evaluated["evaluation_sha256"],
                        "terminal_inventory_mtm_jpy": 0.0,
                        "sealed_signal_or_action_changed": False,
                    })
            new_multiple = 1.0 + net_total / derived.INITIAL_EQUITY_JPY
            sealed = result["periods"]["WALK_FORWARD"][arm]["equity_multiple"]
            prior_value = prior[cycle].get(f"corrected_{arm}")
            cycle_summary["arms"][arm] = {
                "sealed_legacy_multiple": sealed,
                "prior_accounting_only_diagnostic_multiple": prior_value,
                "formal_sign_aware_bbo_fixed_notional_multiple": new_multiple,
                "formal_gross_jpy": gross_total,
                "formal_net_jpy": net_total,
                "formal_total_cost_jpy": cost_total,
                "formal_conversion_bid_ask_width_jpy": conversion_width_total,
                "formal_below_2x": new_multiple < 2.0,
                "diagnostic_only": True,
                "reusable_as_official_seal": False,
            }
        if len(units_by_episode) != episode_count:
            raise MigrationEvidenceError("episode unit identity coverage failed")
        cycle_summary["same_explicit_units_all_arms"] = True
        cycle_summary["terminal_open_inventory"] = 0
        cycle_summary["terminal_inventory_mtm_jpy"] = 0.0
        cycle_summary["hidden_2x_revealed"] = False
        if any(item["formal_sign_aware_bbo_fixed_notional_multiple"] >= 2.0
               for item in cycle_summary["arms"].values()):
            raise MigrationEvidenceError("unexpected 2x diagnostic requires manual audit")
        summary[cycle] = cycle_summary
    rows.sort(key=lambda row: (
        row["cycle_id"], row["episode_id"], row["scenario"]
    ))
    return rows, summary


def _chronology_evidence() -> dict[str, Any]:
    spec = chronology.SessionSpec(
        "LONDON_MIGRATION_FIXTURE", 9 * 60, 9 * 60 + 50, 9 * 60 + 55
    )
    winter = chronology.resolve_completed_m5_session(
        chronology.consecutive_utc_m5("2026-01-15T09:00:00Z", 13),
        "2026-01-15", spec,
    )
    summer = chronology.resolve_completed_m5_session(
        chronology.consecutive_utc_m5("2026-06-15T08:00:00Z", 13),
        "2026-06-15", spec,
    )
    transitions = {
        stamp: chronology.utc_to_london(stamp).as_dict()
        for stamp in (
            "2026-03-29T00:55:00.000000000Z",
            "2026-03-29T01:00:00.000000000Z",
            "2026-10-25T00:55:00.000000000Z",
            "2026-10-25T01:00:00.000000000Z",
        )
    }
    return {
        "winter": winter,
        "summer": summer,
        "transitions": transitions,
        "fixed_utc_hour_used_as_edge_definition": False,
        "classification": "COMMON_CHRONOLOGY_FOUNDATION_NOT_NEW_EDGE",
    }


def _protected_hashes(root: Path) -> dict[str, str]:
    actual = {}
    for relative, expected in PROTECTED.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise MigrationEvidenceError(f"protected sealed artifact changed: {relative}")
        actual[relative] = expected
    return actual


def _source_hashes(root: Path) -> dict[str, str]:
    paths = {
        "policy": POLICY_PATH,
        "accounting_runtime": "jpy_accounting_v2.py",
        "independent_reference": "jpy_accounting_reference_v1.py",
        "london_chronology": "london_session_chronology_v1.py",
        "accounting_tests": "test_jpy_accounting_v2.py",
        "chronology_tests": "test_london_session_chronology_v1.py",
    }
    return {name: sha256_file(root / relative) for name, relative in paths.items()}


def _compute(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    policy = json.loads((root / POLICY_PATH).read_text(encoding="utf-8"))
    if policy.get("classification") != (
        "NON_STRATEGY_RUNTIME_ACCOUNTING_AND_CHRONOLOGY_MIGRATION"
    ) or policy.get("authority") != AUTHORITY:
        raise MigrationEvidenceError("migration policy identity or authority changed")
    if policy["conversion_quote_contract"]["maximum_staleness_seconds"] != (
        accounting.MAX_CONVERSION_STALENESS_SECONDS
    ):
        raise MigrationEvidenceError("conversion staleness policy/runtime mismatch")
    if policy["v42_boundary"]["dst_only_revenue_hypothesis"] != "NO_GO" \
            or policy["holdout"]["state"] != "UNOPENED":
        raise MigrationEvidenceError("migration crossed V42 or holdout boundary")
    fixtures = _fixture_rows()
    diagnostic_rows, diagnostic_summary = _diagnostic_rows_and_summary(root)
    payload = {
        "schema_version": 1,
        "migration_id": policy["migration_id"],
        "classification": policy["classification"],
        "authority": AUTHORITY,
        "source_artifact_sha256": _source_hashes(root),
        "protected_historical_artifact_hashes": _protected_hashes(root),
        "reference_fixture_count": len(fixtures),
        "reference_fixture_parity_passed": all(row["parity"] for row in fixtures),
        "account_currency_midpoint_conversion_used": False,
        "conversion_max_staleness_seconds": accounting.MAX_CONVERSION_STALENESS_SECONDS,
        "sealed_plan_accounting_diagnostic": diagnostic_summary,
        "sealed_signal_or_action_changed": False,
        "historical_seal_rewritten": False,
        "diagnostic_reused_as_official_seal": False,
        "official_strategy_run_performed": False,
        "dst_chronology": _chronology_evidence(),
        "dst_is_revenue_edge": False,
        "v42_dst_only": "NO_GO",
        "current_official_v42_execution_authorized": False,
        "holdout_state": "UNOPENED",
        "external_orders": 0,
        "profit_gate_pass_inferred": False,
        "strategy_adoption_authorized": False,
    }
    return fixtures, diagnostic_rows, payload


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
    )


def expected_artifacts(root: Path) -> tuple[str, str, dict[str, Any]]:
    fixtures, diagnostic_rows, payload = _compute(root)
    fixture_text, diagnostic_text = _jsonl(fixtures), _jsonl(diagnostic_rows)
    payload = {
        **payload,
        "reference_fixture_path": FIXTURE_PATH,
        "reference_fixture_file_sha256": hashlib.sha256(fixture_text.encode()).hexdigest(),
        "diagnostic_rows_path": DIAGNOSTIC_ROWS_PATH,
        "diagnostic_rows": len(diagnostic_rows),
        "diagnostic_rows_file_sha256": hashlib.sha256(diagnostic_text.encode()).hexdigest(),
    }
    payload["audit_sha256"] = embedded_hash(payload, "audit_sha256")
    return fixture_text, diagnostic_text, payload


def build(root: Path) -> dict[str, Any]:
    fixture_text, diagnostic_text, payload = expected_artifacts(root)
    atomic_text(root / FIXTURE_PATH, fixture_text)
    atomic_text(root / DIAGNOSTIC_ROWS_PATH, diagnostic_text)
    atomic_text(
        root / AUDIT_PATH,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return payload


def validate(root: Path) -> dict[str, Any]:
    fixture_text, diagnostic_text, expected = expected_artifacts(root)
    fixture_path = root / FIXTURE_PATH
    diagnostic_path = root / DIAGNOSTIC_ROWS_PATH
    audit_path = root / AUDIT_PATH
    if not all(path.is_file() for path in (fixture_path, diagnostic_path, audit_path)):
        raise MigrationEvidenceError("migration evidence artifact missing")
    if fixture_path.read_text(encoding="utf-8") != fixture_text:
        raise MigrationEvidenceError("reference fixture evidence changed")
    if diagnostic_path.read_text(encoding="utf-8") != diagnostic_text:
        raise MigrationEvidenceError("accounting diagnostic rows changed")
    actual = json.loads(audit_path.read_text(encoding="utf-8"))
    if actual != expected or actual.get("audit_sha256") != embedded_hash(actual, "audit_sha256"):
        raise MigrationEvidenceError("migration audit evidence changed")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"), nargs="?", default="build")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    payload = build(args.root) if args.command == "build" else validate(args.root)
    print(json.dumps({
        "migration_id": payload["migration_id"],
        "audit_sha256": payload["audit_sha256"],
        "reference_fixture_parity_passed": payload["reference_fixture_parity_passed"],
        "v42_dst_only": payload["v42_dst_only"],
        "official_strategy_run_performed": payload["official_strategy_run_performed"],
        "holdout_state": payload["holdout_state"],
        "external_orders": payload["external_orders"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
