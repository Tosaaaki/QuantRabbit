from __future__ import annotations

import ast
import builtins
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import inspect
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import pytest

import paper_research_double_entry_reference_v2 as reference


START_NS = 1_767_225_600_000_000_000
DAY_NS = 86_400_000_000_000
END_NS = 1_769_904_000_000_000_000
MARCH_START_NS = START_NS + 59 * DAY_NS
ZERO_SHA256 = "0" * 64
EXPECTED_ENGINE_ID = "EVENT_SOURCED_DOUBLE_ENTRY_REFERENCE_V1"
ARTIFACT_KEYS = frozenset({
    "source_blob",
    "source_manifest",
    "proposal",
    "execution_policy",
    "inventory_policy",
    "accounting_policy",
    "evaluation_policy",
    "instrument_registry",
    "authority_policy",
})
EXPECTED_RESULT_KEYS = frozenset({
    "engine_id",
    "input_root_sha256",
    "ledger_bytes",
    "ledger_row_count",
    "ledger_terminal_hash",
    "oracle_metrics",
    "proposal_provenance_root_sha256",
    "journal_root_sha256",
    "journal_transaction_count",
    "all_transactions_balanced",
    "economic_projection_sha256",
})
ARMS = ("RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result.pop(field, None)
    result[field] = _digest(_canonical(result))
    return result


def _json_artifact(value: Mapping[str, Any]) -> bytes:
    return _canonical(value) + b"\n"


@dataclass(frozen=True)
class CanonicalFixture:
    artifacts: Mapping[str, bytes]
    source_rows: tuple[Mapping[str, Any], ...]
    enriched_rows: tuple[Mapping[str, Any], ...]
    values: Mapping[str, Mapping[str, Any]]


def _build_canonical_fixture() -> CanonicalFixture:
    # These are deliberately literal market observations, not copied from or
    # produced by the Oracle, verifier, runner, or an accounting implementation.
    source_rows: tuple[dict[str, Any], ...] = (
        {
            "schema_version": 1,
            "provider_id": "DOUBLE_ENTRY_GOLDEN",
            "instrument": "USD_JPY",
            "bid_ticks": 9_999,
            "ask_ticks": 10_001,
            "tick_scale": 100,
            "source_ts_ns": START_NS,
            "arrival_ts_ns": START_NS,
            "provider_event_id": "de-g1",
            "sequence": 1,
            "heartbeat": False,
            "quality_flags": [],
        },
        {
            "schema_version": 1,
            "provider_id": "DOUBLE_ENTRY_GOLDEN",
            "instrument": "USD_JPY",
            "bid_ticks": 9_999,
            "ask_ticks": 10_001,
            "tick_scale": 100,
            "source_ts_ns": START_NS + 1_000_000_000,
            "arrival_ts_ns": START_NS + 1_000_000_000,
            "provider_event_id": "de-g2",
            "sequence": 2,
            "heartbeat": False,
            "quality_flags": [],
        },
        {
            "schema_version": 1,
            "provider_id": "DOUBLE_ENTRY_GOLDEN",
            "instrument": "USD_JPY",
            "bid_ticks": 10_099,
            "ask_ticks": 10_101,
            "tick_scale": 100,
            "source_ts_ns": START_NS + DAY_NS + 1_000_000_000,
            "arrival_ts_ns": START_NS + DAY_NS + 1_000_000_000,
            "provider_event_id": "de-g3",
            "sequence": 3,
            "heartbeat": False,
            "quality_flags": [],
        },
    )
    source_lines = tuple(_canonical(row) + b"\n" for row in source_rows)
    source_blob = b"".join(source_lines)
    prefix = ZERO_SHA256
    enriched: list[dict[str, Any]] = []
    for row, raw_line in zip(source_rows, source_lines):
        event_hash = _digest(raw_line)
        prefix = _digest(_canonical({
            "previous_hash": prefix,
            "source_event_sha256": event_hash,
        }))
        enriched.append({
            **row,
            "source_event_sha256": event_hash,
            "source_prefix_root_sha256": prefix,
        })

    registry = _seal({
        "schema_version": 1,
        "registry_id": "FROZEN_FX_INSTRUMENT_REGISTRY_V1",
        "instruments": {"USD_JPY": {"pip_ticks": 1, "price_scale": 100}},
    }, "registry_sha256")
    source_manifest = _seal({
        "schema_version": 2,
        "source_bytes_sha256": _digest(source_blob),
        "source_size_bytes": len(source_blob),
        "event_count": 3,
        "first_source_ts_ns": START_NS,
        "last_source_ts_ns": START_NS + DAY_NS + 1_000_000_000,
        "provider_allowlist": ["DOUBLE_ENTRY_GOLDEN"],
        "instrument_registry_sha256": registry["registry_sha256"],
        "stream_policies": [{
            "provider_id": "DOUBLE_ENTRY_GOLDEN",
            "instrument": "USD_JPY",
            "sequence_required": True,
            "first_sequence": 1,
            "last_sequence": 3,
            "event_count": 3,
            "max_source_gap_ns": DAY_NS,
            "max_arrival_gap_ns": DAY_NS,
        }],
        "lossless": True,
    }, "manifest_sha256")
    provenance = {
        "detector_code_sha256": "1" * 64,
        "detector_policy_sha256": "2" * 64,
        "generator_policy_sha256": "3" * 64,
        "source_acquisition_contract_sha256": "4" * 64,
    }
    proposal = _seal({
        "schema_version": 2,
        "candidate_key": "DOUBLE-ENTRY-GOLDEN-USDJPY-LONG",
        "provenance": provenance,
        "rows": [{
            "proposal_ordinal": 1,
            "decision_source_ts_ns": START_NS,
            "decision_arrival_ts_ns": START_NS,
            "available_at_ns": START_NS,
            "decision_source_event_sha256": enriched[0]["source_event_sha256"],
            "completed_data_watermark_source_ts_ns": START_NS,
            "completed_data_prefix_root_sha256": enriched[0][
                "source_prefix_root_sha256"
            ],
            "instrument": "USD_JPY",
            "direction": 1,
            "notional_jpy_micros": 100_000_000,
            "max_age_ns": DAY_NS,
            "worker_key": "DOUBLE_ENTRY_FIXED",
            "action": "ENTER",
        }],
    }, "proposal_sha256")
    execution = _seal({
        "schema_version": 2,
        "policy_id": "FROZEN_EXECUTION_POLICY_V2",
        "arms": {
            "RAW_SIGNAL": {
                "latency_ns": 0,
                "slippage_micropips_per_side": 0,
                "commission_ppm_per_side": 0,
                "financing_ppm_per_day": 0,
                "raw_mid": True,
            },
            "EXECUTABLE_BASE": {
                "latency_ns": 0,
                "slippage_micropips_per_side": 0,
                "commission_ppm_per_side": 10,
                "financing_ppm_per_day": 5,
                "raw_mid": False,
            },
            "ADVERSE_STRESS": {
                "latency_ns": 0,
                "slippage_micropips_per_side": 1_000_000,
                "commission_ppm_per_side": 20,
                "financing_ppm_per_day": 10,
                "raw_mid": False,
            },
        },
        "max_trade_quote_staleness_ns": DAY_NS,
    }, "execution_policy_sha256")
    inventory = _seal({
        "schema_version": 2,
        "policy_id": "FROZEN_INVENTORY_POLICY_V2",
        "max_gross_notional_jpy_micros": 1_000_000_000,
        "max_currency_notional_jpy_micros": 1_000_000_000,
        "max_open_positions": 1,
        "same_pair_collision": "REJECT_NEW",
        "terminal_liquidation": True,
    }, "inventory_policy_sha256")
    accounting = _seal({
        "schema_version": 2,
        "policy_id": "FROZEN_ACCOUNTING_POLICY_V2",
        "jpy_micros_per_yen": 1_000_000,
        "base_microunits_per_unit": 1_000_000,
        "max_conversion_staleness_ns": DAY_NS,
        "supported_quote_currencies": ["CAD", "CHF", "JPY", "USD"],
        "asset_conversion_side": "BID",
        "liability_conversion_side": "ASK",
        "positive_cost_rounding": "CEILING",
    }, "accounting_policy_sha256")
    evaluation = _seal({
        "schema_version": 2,
        "policy_id": "FROZEN_EVALUATION_POLICY_V2",
        "period_start_ts_ns": START_NS,
        "period_end_ts_ns": END_NS,
        "initial_equity_jpy_micros": 1_000_000_000,
        "margin_notional_cap_jpy_micros": 1_000_000_000,
        "margin_rate_bps": 500,
        "max_gross_to_equity_bps": 20_000,
        "cvar_tail_bps": 500,
        "cluster_window_ns": 1_000_000_000,
        "full_month_ids": ["2026-01"],
        "holdout_state": "UNOPENED",
    }, "evaluation_policy_sha256")
    authority = _seal({
        "schema_version": 2,
        "policy_id": "FROZEN_PAPER_AUTHORITY_V1",
        "paper_only": True,
        "live_authority": False,
        "broker_account_access": False,
        "credential_access": False,
        "order_endpoint": False,
        "external_orders": 0,
        "deploy": False,
        "external_config_mutation": False,
    }, "authority_policy_sha256")
    values = {
        "source_manifest": source_manifest,
        "proposal": proposal,
        "execution_policy": execution,
        "inventory_policy": inventory,
        "accounting_policy": accounting,
        "evaluation_policy": evaluation,
        "instrument_registry": registry,
        "authority_policy": authority,
    }
    artifacts = {
        "source_blob": source_blob,
        **{label: _json_artifact(value) for label, value in values.items()},
    }
    assert set(artifacts) == set(ARTIFACT_KEYS)
    return CanonicalFixture(
        artifacts=MappingProxyType(artifacts),
        source_rows=tuple(MappingProxyType(row) for row in source_rows),
        enriched_rows=tuple(MappingProxyType(row) for row in enriched),
        values=MappingProxyType({
            key: MappingProxyType(value) for key, value in values.items()
        }),
    )


@pytest.fixture(scope="module")
def canonical_fixture() -> CanonicalFixture:
    return _build_canonical_fixture()


@dataclass(frozen=True)
class MatrixFixture:
    artifacts: Mapping[str, bytes]
    enriched_by_tag: Mapping[str, Mapping[str, Any]]


def _build_matrix_fixture(
    fixture: CanonicalFixture,
    *,
    events: tuple[Mapping[str, Any], ...],
    proposals: tuple[Mapping[str, Any], ...],
    registry: Mapping[str, Mapping[str, int]],
    mutate_execution: Callable[[dict[str, Any]], None] | None = None,
    mutate_inventory: Callable[[dict[str, Any]], None] | None = None,
    mutate_accounting: Callable[[dict[str, Any]], None] | None = None,
    mutate_evaluation: Callable[[dict[str, Any]], None] | None = None,
) -> MatrixFixture:
    """Seal test-local raw observations without consulting a producer engine."""
    provider = "DOUBLE_ENTRY_MATRIX"
    stream_sequence: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    tags: list[str] = []
    for event in sorted(
        events,
        key=lambda item: (
            item["arrival_ts_ns"],
            item["source_ts_ns"],
            item["instrument"],
            item["tag"],
        ),
    ):
        instrument = event["instrument"]
        stream_sequence[instrument] += 1
        tags.append(event["tag"])
        rows.append({
            "schema_version": 1,
            "provider_id": provider,
            "instrument": instrument,
            "bid_ticks": event["bid_ticks"],
            "ask_ticks": event["ask_ticks"],
            "tick_scale": registry[instrument]["price_scale"],
            "source_ts_ns": event["source_ts_ns"],
            "arrival_ts_ns": event["arrival_ts_ns"],
            "provider_event_id": event["tag"],
            "sequence": stream_sequence[instrument],
            "heartbeat": False,
            "quality_flags": [],
        })
    raw_lines = tuple(_canonical(row) + b"\n" for row in rows)
    source_blob = b"".join(raw_lines)
    prefix = ZERO_SHA256
    enriched_by_tag: dict[str, dict[str, Any]] = {}
    enriched_rows: list[dict[str, Any]] = []
    for tag, row, raw_line in zip(tags, rows, raw_lines):
        event_hash = _digest(raw_line)
        prefix = _digest(_canonical({
            "previous_hash": prefix,
            "source_event_sha256": event_hash,
        }))
        enriched = {
            **row,
            "source_event_sha256": event_hash,
            "source_prefix_root_sha256": prefix,
        }
        enriched_rows.append(enriched)
        enriched_by_tag[tag] = enriched

    registry_payload = _seal({
        "schema_version": 1,
        "registry_id": "FROZEN_FX_INSTRUMENT_REGISTRY_V1",
        "instruments": {
            instrument: dict(registry[instrument])
            for instrument in sorted(registry)
        },
    }, "registry_sha256")
    stream_policies: list[dict[str, Any]] = []
    for instrument in sorted(stream_sequence):
        stream = [row for row in rows if row["instrument"] == instrument]
        source_gaps = [
            later["source_ts_ns"] - earlier["source_ts_ns"]
            for earlier, later in zip(stream, stream[1:])
        ]
        arrival_gaps = [
            later["arrival_ts_ns"] - earlier["arrival_ts_ns"]
            for earlier, later in zip(stream, stream[1:])
        ]
        stream_policies.append({
            "provider_id": provider,
            "instrument": instrument,
            "sequence_required": True,
            "first_sequence": 1,
            "last_sequence": len(stream),
            "event_count": len(stream),
            "max_source_gap_ns": max(source_gaps, default=1),
            "max_arrival_gap_ns": max(arrival_gaps, default=1),
        })
    source_manifest = _seal({
        "schema_version": 2,
        "source_bytes_sha256": _digest(source_blob),
        "source_size_bytes": len(source_blob),
        "event_count": len(rows),
        "first_source_ts_ns": min(row["source_ts_ns"] for row in rows),
        "last_source_ts_ns": max(row["source_ts_ns"] for row in rows),
        "provider_allowlist": [provider],
        "instrument_registry_sha256": registry_payload["registry_sha256"],
        "stream_policies": stream_policies,
        "lossless": True,
    }, "manifest_sha256")
    proposal_rows: list[dict[str, Any]] = []
    for ordinal, spec in enumerate(proposals, 1):
        decision = enriched_by_tag[spec["decision_tag"]]
        available = [
            row
            for row in enriched_rows
            if row["arrival_ts_ns"] <= decision["arrival_ts_ns"]
        ]
        proposal_rows.append({
            "proposal_ordinal": ordinal,
            "decision_source_ts_ns": decision["source_ts_ns"],
            "decision_arrival_ts_ns": decision["arrival_ts_ns"],
            "available_at_ns": decision["arrival_ts_ns"],
            "decision_source_event_sha256": decision["source_event_sha256"],
            "completed_data_watermark_source_ts_ns": max(
                row["source_ts_ns"] for row in available
            ),
            "completed_data_prefix_root_sha256": available[-1][
                "source_prefix_root_sha256"
            ],
            "instrument": spec["instrument"],
            "direction": spec["direction"],
            "notional_jpy_micros": spec.get("notional_jpy_micros", 100_000_000),
            "max_age_ns": spec.get("max_age_ns", DAY_NS),
            "worker_key": spec.get("worker_key", f"MATRIX-{ordinal}"),
            "action": "ENTER",
        })
    proposal_payload = _seal({
        "schema_version": 2,
        "candidate_key": "DOUBLE-ENTRY-MATRIX",
        "provenance": dict(fixture.values["proposal"]["provenance"]),
        "rows": proposal_rows,
    }, "proposal_sha256")

    values: dict[str, dict[str, Any]] = {
        "execution_policy": deepcopy(dict(fixture.values["execution_policy"])),
        "inventory_policy": deepcopy(dict(fixture.values["inventory_policy"])),
        "accounting_policy": deepcopy(dict(fixture.values["accounting_policy"])),
        "evaluation_policy": deepcopy(dict(fixture.values["evaluation_policy"])),
        "authority_policy": deepcopy(dict(fixture.values["authority_policy"])),
    }
    for label, hash_field, mutate in (
        ("execution_policy", "execution_policy_sha256", mutate_execution),
        ("inventory_policy", "inventory_policy_sha256", mutate_inventory),
        ("accounting_policy", "accounting_policy_sha256", mutate_accounting),
        ("evaluation_policy", "evaluation_policy_sha256", mutate_evaluation),
    ):
        values[label].pop(hash_field, None)
        if mutate is not None:
            mutate(values[label])
        values[label] = _seal(values[label], hash_field)
    artifacts = {
        "source_blob": source_blob,
        "source_manifest": _json_artifact(source_manifest),
        "proposal": _json_artifact(proposal_payload),
        "instrument_registry": _json_artifact(registry_payload),
        **{
            label: _json_artifact(value)
            for label, value in values.items()
        },
    }
    assert set(artifacts) == set(ARTIFACT_KEYS)
    return MatrixFixture(
        artifacts=MappingProxyType(artifacts),
        enriched_by_tag=MappingProxyType({
            tag: MappingProxyType(value)
            for tag, value in enriched_by_tag.items()
        }),
    )


# Independently reviewed economic literals.  Hashing and serialization below
# are mechanical; none of these economic values are obtained from production.
LITERAL_ECONOMICS: Mapping[str, Mapping[str, int]] = MappingProxyType({
    "RAW_SIGNAL": MappingProxyType({
        "entry_numerator": 20_000_000_000,
        "entry_denominator": 200_000_000,
        "exit_numerator": 20_200_000_000,
        "exit_denominator": 200_000_000,
        "units_micros": 1_000_000,
        "target_notional": 100_000_000,
        "filled_notional": 100_000_000,
        "financing_basis_notional": 100_000_000,
        "exit_notional": 101_000_000,
        "arm_units_common_gross": 1_000_000,
        "executable": 1_000_000,
        "fill_sizing_drag": 0,
        "execution_drag": 0,
        "commission": 0,
        "financing": 0,
        "cost": 0,
        "net": 1_000_000,
        "economic_net_numerator": 1_000_000,
        "economic_net_denominator": 1,
        "entry_exposure": 100_000_000,
        "base_node_exposure": 99_990_000,
        "quote_node_exposure": -100_000_000,
        "entry_equity": 1_000_000_000,
        "required_margin": 5_000_000,
        "free_margin": 995_000_000,
        "ending_equity": 1_001_000_000,
        "drawdown": 0,
    }),
    "EXECUTABLE_BASE": MappingProxyType({
        "entry_numerator": 10_001_000_000,
        "entry_denominator": 100_000_000,
        "exit_numerator": 10_099_000_000,
        "exit_denominator": 100_000_000,
        "units_micros": 999_900,
        "target_notional": 100_000_000,
        "filled_notional": 99_999_999,
        "financing_basis_notional": 99_999_999,
        "exit_notional": 100_979_901,
        "arm_units_common_gross": 999_900,
        "executable": 979_902,
        "fill_sizing_drag": 100,
        "execution_drag": 19_998,
        "commission": 2_010,
        "financing": 500,
        "cost": 22_608,
        "net": 977_392,
        "economic_net_numerator": 195_478_440_201,
        "economic_net_denominator": 200_000,
        "entry_exposure": 99_980_001,
        "base_node_exposure": 99_980_001,
        "quote_node_exposure": -99_980_001,
        "entry_equity": 999_978_002,
        "required_margin": 4_999_001,
        "free_margin": 994_979_001,
        "ending_equity": 1_000_977_392,
        "drawdown": 21_998,
    }),
    "ADVERSE_STRESS": MappingProxyType({
        "entry_numerator": 10_002_000_000,
        "entry_denominator": 100_000_000,
        "exit_numerator": 10_098_000_000,
        "exit_denominator": 100_000_000,
        "units_micros": 999_800,
        "target_notional": 100_000_000,
        "filled_notional": 99_999_996,
        "financing_basis_notional": 99_999_996,
        "exit_notional": 100_959_804,
        "arm_units_common_gross": 999_800,
        "executable": 959_808,
        "fill_sizing_drag": 200,
        "execution_drag": 39_992,
        "commission": 4_020,
        "financing": 1_000,
        "cost": 45_212,
        "net": 954_788,
        "economic_net_numerator": 23_869_720_101,
        "economic_net_denominator": 25_000,
        "entry_exposure": 99_960_004,
        "base_node_exposure": 99_970_002,
        "quote_node_exposure": -99_960_004,
        "entry_equity": 999_956_008,
        "required_margin": 4_998_001,
        "free_margin": 994_958_007,
        "ending_equity": 1_000_954_788,
        "drawdown": 43_992,
    }),
})


def _ratio_text(numerator: int, denominator: int) -> str:
    scaled = numerator * 10**18 // denominator
    sign = "-" if scaled < 0 else ""
    magnitude = abs(scaled)
    return f"{sign}{magnitude // 10**18}.{magnitude % 10**18:018d}"


def _outward_ratio_text(numerator: int, denominator: int) -> str:
    scaled = (numerator * 10**18 + denominator - 1) // denominator
    return f"{scaled // 10**18}.{scaled % 10**18:018d}"


def _literal_identities(fixture: CanonicalFixture) -> tuple[str, str]:
    proposal = fixture.values["proposal"]
    row = proposal["rows"][0]
    provenance = proposal["provenance"]
    signal_id = _digest(_canonical({
        "candidate_key": proposal["candidate_key"],
        "proposal_ordinal": 1,
        "decision_source_ts_ns": row["decision_source_ts_ns"],
        "decision_arrival_ts_ns": row["decision_arrival_ts_ns"],
        "decision_source_event_sha256": row["decision_source_event_sha256"],
        "completed_data_prefix_root_sha256": row[
            "completed_data_prefix_root_sha256"
        ],
        "instrument": "USD_JPY",
        "direction": 1,
        "notional_jpy_micros": 100_000_000,
        "max_age_ns": DAY_NS,
        "worker_key": "DOUBLE_ENTRY_FIXED",
        "detector_code_sha256": provenance["detector_code_sha256"],
        "detector_policy_sha256": provenance["detector_policy_sha256"],
        "generator_policy_sha256": provenance["generator_policy_sha256"],
    }))
    economic_lot_id = _digest(_canonical({
        "candidate_key": proposal["candidate_key"],
        "decision_source_ts_ns": row["decision_source_ts_ns"],
        "decision_arrival_ts_ns": row["decision_arrival_ts_ns"],
        "decision_source_event_sha256": row["decision_source_event_sha256"],
        "completed_data_prefix_root_sha256": row[
            "completed_data_prefix_root_sha256"
        ],
        "instrument": "USD_JPY",
        "direction": 1,
        "target_notional_jpy_micros": 100_000_000,
        "max_age_ns": DAY_NS,
        "worker_key": "DOUBLE_ENTRY_FIXED",
        "detector_code_sha256": provenance["detector_code_sha256"],
        "detector_policy_sha256": provenance["detector_policy_sha256"],
        "generator_policy_sha256": provenance["generator_policy_sha256"],
    }))
    return signal_id, economic_lot_id


def _literal_ledger_rows(fixture: CanonicalFixture) -> list[dict[str, Any]]:
    signal_id, economic_lot_id = _literal_identities(fixture)
    entry = fixture.enriched_rows[1]
    exit_event = fixture.enriched_rows[2]
    execution_hash = fixture.values["execution_policy"]["execution_policy_sha256"]

    def source_reference(event: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "provider_id": "DOUBLE_ENTRY_GOLDEN",
            "source_event_sha256": event["source_event_sha256"],
            "source_ts_ns": event["source_ts_ns"],
            "arrival_ts_ns": event["arrival_ts_ns"],
            "execution_policy_sha256": execution_hash,
        }

    dispositions: list[dict[str, Any]] = []
    for arm in ARMS:
        value = LITERAL_ECONOMICS[arm]
        dispositions.append({
            "record_type": "ORACLE_DISPOSITION",
            "arm": arm,
            "signal_id": signal_id,
            "proposal_ordinal": 1,
            "instrument": "USD_JPY",
            "direction": 1,
            "status": "FILLED_CLOSED",
            "entry_disposition": "FILLED",
            "exit_disposition": "FINITE_MAX_AGE",
            "action_transitions": ["ENTER", "EXIT"],
            "notional_jpy_micros": 100_000_000,
            "target_notional_jpy_micros": value["target_notional"],
            "filled_notional_jpy_micros": value["filled_notional"],
            "financing_basis_notional_jpy_micros": value[
                "financing_basis_notional"
            ],
            "marked_or_exit_notional_jpy_micros": value["exit_notional"],
            "exit_notional_jpy_micros": value["exit_notional"],
            "units_micros": value["units_micros"],
            "economic_lot_id": economic_lot_id,
            "common_entry_source_event_sha256": entry["source_event_sha256"],
            "common_exit_source_event_sha256": exit_event["source_event_sha256"],
            "common_gross_pnl_jpy_micros": 1_000_000,
            "arm_units_common_gross_pnl_jpy_micros": value[
                "arm_units_common_gross"
            ],
            "entry_price_numerator": value["entry_numerator"],
            "entry_price_denominator": value["entry_denominator"],
            "exit_price_numerator": value["exit_numerator"],
            "exit_price_denominator": value["exit_denominator"],
            "entry_source_event_sha256": entry["source_event_sha256"],
            "entry_source_ts_ns": entry["source_ts_ns"],
            "entry_arrival_ts_ns": entry["arrival_ts_ns"],
            "exit_source_event_sha256": exit_event["source_event_sha256"],
            "exit_source_ts_ns": exit_event["source_ts_ns"],
            "exit_arrival_ts_ns": exit_event["arrival_ts_ns"],
            "elapsed_ns": DAY_NS,
            "executable_pnl_before_direct_cost_jpy_micros": value["executable"],
            "fill_sizing_drag_jpy_micros": value["fill_sizing_drag"],
            "latency_spread_slippage_drag_jpy_micros": value["execution_drag"],
            "commission_jpy_micros": value["commission"],
            "financing_jpy_micros": value["financing"],
            "realized_cost_jpy_micros": value["cost"],
            "admission_opportunity_drag_jpy_micros": 0,
            "net_pnl_jpy_micros": value["net"],
            "economic_net_pnl_jpy_micros_numerator": value[
                "economic_net_numerator"
            ],
            "economic_net_pnl_jpy_micros_denominator": value[
                "economic_net_denominator"
            ],
            "signed_currency_exposure_after_entry_jpy_micros": {
                "JPY": value["quote_node_exposure"],
                "USD": value["base_node_exposure"],
            },
            "gross_open_notional_after_entry_jpy_micros": value["entry_exposure"],
            "marked_equity_after_entry_jpy_micros": value["entry_equity"],
            "required_margin_after_entry_jpy_micros": value["required_margin"],
            "free_margin_after_entry_jpy_micros": value["free_margin"],
            "entry_source_reference": source_reference(entry),
            "exit_source_reference": source_reference(exit_event),
            "terminal_inventory_mtm_jpy_micros": 0,
            "external_order_count": 0,
        })
    previous = ZERO_SHA256
    rows: list[dict[str, Any]] = []
    for sequence, disposition in enumerate(dispositions, 1):
        row = {
            "ledger_schema_version": 2,
            "ledger_sequence": sequence,
            "previous_hash": previous,
            **disposition,
        }
        row["record_hash"] = _digest(_canonical(row))
        previous = row["record_hash"]
        rows.append(row)
    return rows


def _literal_ledger_bytes(fixture: CanonicalFixture) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in _literal_ledger_rows(fixture))


def _literal_metrics(fixture: CanonicalFixture) -> dict[str, Any]:
    signal_id, economic_lot_id = _literal_identities(fixture)
    signal_set_hash = _digest(_canonical([signal_id]))
    lot_set_hash = _digest(_canonical([economic_lot_id]))
    time_bucket = 1_767_225_601
    cluster_id = _digest(_canonical({
        "time_bucket": time_bucket,
        "currency_nodes": ["JPY", "USD"],
    }))
    arm_metrics: dict[str, Any] = {}
    for arm in ARMS:
        value = LITERAL_ECONOMICS[arm]
        arm_metrics[arm] = {
            "proposal_count": 1,
            "executed_count": 1,
            "disposition_counts": {"FILLED_CLOSED": 1},
            "signal_id_set_sha256": signal_set_hash,
            "common_gross_pnl_jpy_micros": 1_000_000,
            "realized_cost_jpy_micros": value["cost"],
            "fill_sizing_drag_jpy_micros": value["fill_sizing_drag"],
            "latency_spread_slippage_drag_jpy_micros": value["execution_drag"],
            "direct_commission_financing_cost_jpy_micros": (
                value["commission"] + value["financing"]
            ),
            "admission_opportunity_drag_jpy_micros": 0,
            "total_execution_and_admission_drag_jpy_micros": value["cost"],
            "net_pnl_jpy_micros": value["net"],
            "ending_equity_jpy_micros": value["ending_equity"],
            "ending_equity_multiple": _ratio_text(
                value["ending_equity"], 1_000_000_000
            ),
            "direction_accuracy": "1.000000000000000000",
            "max_drawdown_jpy_micros": value["drawdown"],
            "max_drawdown_ratio": _outward_ratio_text(
                value["drawdown"], 1_000_000_000
            ),
            "cvar_tail_bps": 500,
            "cluster_cvar_jpy_micros": value["net"],
            "cluster_cvar_return": _ratio_text(
                value["economic_net_numerator"],
                value["economic_net_denominator"] * 1_000_000_000,
            ),
            "currency_time_cluster_n_eff": 1,
            "currency_time_cluster_observations": [{
                "cluster_id": cluster_id,
                "time_bucket": time_bucket,
                "currency_nodes": ["JPY", "USD"],
                "source_signal_set_sha256": lot_set_hash,
                "ledger_net_pnl_jpy_micros": value["net"],
                "cluster_risk_net_pnl_jpy_micros": value["net"],
                "signed_return": _ratio_text(value["net"], 1_000_000_000),
            }],
            "monthly": [{
                "month_id": "2026-01",
                "comparable_full_month": True,
                "segment_start_ts_ns": START_NS,
                "segment_end_ts_ns": END_NS,
                "start_equity_jpy_micros": 1_000_000_000,
                "end_equity_jpy_micros": value["ending_equity"],
                "equity_multiple": _ratio_text(
                    value["ending_equity"], 1_000_000_000
                ),
                "equity_multiple_status": "DEFINED",
                "ruin_observed": False,
            }],
            "max_gross_notional_jpy_micros": value["entry_exposure"],
            "minimum_marked_equity_jpy_micros": value["entry_equity"],
            "maximum_required_margin_jpy_micros": value["required_margin"],
            "minimum_free_margin_jpy_micros": value["free_margin"],
            "margin_guard_pass": True,
            "terminal_open_positions": 0,
            "terminal_inventory_mtm_jpy_micros": 0,
        }
    metrics: dict[str, Any] = {
        "schema_version": 2,
        "initial_equity_jpy_micros": 1_000_000_000,
        "same_signal_ids_all_arms": True,
        "all_proposals_have_all_arm_dispositions": True,
        "common_gross_reference_shared": True,
        "arms": arm_metrics,
        "external_orders": 0,
        "terminal_inventory_mtm_jpy_micros": 0,
    }
    metrics["metrics_sha256"] = _digest(_canonical(metrics))
    return metrics


def _literal_journal_transactions(
    fixture: CanonicalFixture,
) -> list[dict[str, Any]]:
    entry = fixture.enriched_rows[1]
    exit_event = fixture.enriched_rows[2]
    transactions: list[dict[str, Any]] = []

    def post(
        arm: str,
        event_kind: str,
        event_id: str,
        arrival_ns: int,
        source_hash: str,
        postings: tuple[tuple[str, Fraction], ...],
    ) -> None:
        canonical_postings = tuple(
            (account, amount) for account, amount in postings if amount
        )
        assert len(canonical_postings) >= 2
        assert sum(
            (amount for _, amount in canonical_postings), Fraction()
        ) == 0
        transactions.append({
            "arrival_ts_ns": arrival_ns,
            "arm": arm,
            "proposal_ordinal": 1,
            "event_kind": event_kind,
            "event_id": event_id,
            "source_event_sha256": source_hash,
            "postings": canonical_postings,
        })

    for arm in ARMS:
        value = LITERAL_ECONOMICS[arm]
        basis = Fraction(value["filled_notional"])
        open_id = f"{arm}:1:OPEN:{entry['source_event_sha256']}"
        close_prefix = f"{arm}:1:CLOSE:{exit_event['arrival_ts_ns']}"
        post(
            arm,
            "POSITION_OPEN",
            open_id,
            entry["arrival_ts_ns"],
            entry["source_event_sha256"],
            (("POSITION_BASIS", basis), ("POSITION_CONTROL", -basis)),
        )
        if arm == "EXECUTABLE_BASE":
            entry_commission = Fraction(99_999_999, 100_000)
            initial_mark = Fraction(-19_998)
            exit_commission = Fraction(100_979_901, 100_000)
            financing = Fraction(99_999_999, 200_000)
        elif arm == "ADVERSE_STRESS":
            entry_commission = Fraction(24_999_999, 12_500)
            initial_mark = Fraction(-39_992)
            exit_commission = Fraction(25_239_951, 12_500)
            financing = Fraction(24_999_999, 25_000)
        else:
            entry_commission = Fraction()
            initial_mark = Fraction()
            exit_commission = Fraction()
            financing = Fraction()
        if entry_commission:
            post(
                arm,
                "ENTRY_COMMISSION",
                f"{arm}:1:ENTRY_COMMISSION",
                entry["arrival_ts_ns"],
                entry["source_event_sha256"],
                (
                    ("COMMISSION_EXPENSE", entry_commission),
                    ("SETTLEMENT_CASH", -entry_commission),
                ),
            )
        if initial_mark:
            post(
                arm,
                "MARK_DELTA",
                f"{arm}:1:MARK:{entry['arrival_ts_ns']}",
                entry["arrival_ts_ns"],
                entry["source_event_sha256"],
                (
                    ("UNREALIZED_ASSET", initial_mark),
                    ("UNREALIZED_PNL", -initial_mark),
                ),
            )
            post(
                arm,
                "REVERSE_UNREALIZED",
                close_prefix + ":REVERSE_MARK",
                exit_event["arrival_ts_ns"],
                exit_event["source_event_sha256"],
                (
                    ("UNREALIZED_ASSET", -initial_mark),
                    ("UNREALIZED_PNL", initial_mark),
                ),
            )
        post(
            arm,
            "POSITION_CLOSE",
            close_prefix + ":BASIS",
            exit_event["arrival_ts_ns"],
            exit_event["source_event_sha256"],
            (("POSITION_BASIS", -basis), ("POSITION_CONTROL", basis)),
        )
        realized = Fraction(value["executable"])
        post(
            arm,
            "REALIZE_TRADING_PNL",
            close_prefix + ":REALIZE",
            exit_event["arrival_ts_ns"],
            exit_event["source_event_sha256"],
            (("SETTLEMENT_CASH", realized), ("REALIZED_TRADING_PNL", -realized)),
        )
        if exit_commission:
            post(
                arm,
                "EXIT_COMMISSION",
                close_prefix + ":EXIT_COMMISSION",
                exit_event["arrival_ts_ns"],
                exit_event["source_event_sha256"],
                (
                    ("COMMISSION_EXPENSE", exit_commission),
                    ("SETTLEMENT_CASH", -exit_commission),
                ),
            )
        if financing:
            post(
                arm,
                "FINANCING",
                close_prefix + ":FINANCING",
                exit_event["arrival_ts_ns"],
                exit_event["source_event_sha256"],
                (
                    ("FINANCING_EXPENSE", financing),
                    ("SETTLEMENT_CASH", -financing),
                ),
            )
        direct_cost = value["commission"] + value["financing"]
        attribution = (
            ("COMMON_GROSS_REFERENCE", Fraction(1_000_000)),
            ("FILL_SIZING_DRAG", Fraction(-value["fill_sizing_drag"])),
            (
                "LATENCY_SPREAD_SLIPPAGE_DRAG",
                Fraction(-value["execution_drag"]),
            ),
            ("DIRECT_COST", Fraction(-direct_cost)),
            ("NET_PNL_CONTROL", Fraction(-value["net"])),
        )
        post(
            arm,
            "ATTRIBUTION_CONTROL",
            close_prefix + ":ATTRIBUTION",
            exit_event["arrival_ts_ns"],
            exit_event["source_event_sha256"],
            attribution,
        )
    return transactions


def _literal_journal_root(fixture: CanonicalFixture) -> str:
    previous = ZERO_SHA256
    for sequence, transaction in enumerate(
        _literal_journal_transactions(fixture), 1
    ):
        payload = {
            "journal_schema_version": 1,
            "sequence": sequence,
            "previous_hash": previous,
            "arrival_ts_ns": transaction["arrival_ts_ns"],
            "arm": transaction["arm"],
            "proposal_ordinal": transaction["proposal_ordinal"],
            "event_kind": transaction["event_kind"],
            "event_id": transaction["event_id"],
            "source_event_sha256": transaction["source_event_sha256"],
            "postings": [
                {
                    "account": account,
                    "amount_numerator": amount.numerator,
                    "amount_denominator": amount.denominator,
                }
                for account, amount in transaction["postings"]
            ],
        }
        previous = _digest(_canonical(payload))
    return previous


def _literal_result_roots(fixture: CanonicalFixture) -> dict[str, str]:
    raw_hashes = {
        key: _digest(fixture.artifacts[key]) for key in sorted(ARTIFACT_KEYS)
    }
    input_root = _digest(_canonical({
        "artifact_sha256": dict(sorted(raw_hashes.items())),
    }))
    row = fixture.values["proposal"]["rows"][0]
    proposal_root = _digest(_canonical({
        "provenance": dict(fixture.values["proposal"]["provenance"]),
        "rows": [{
            "proposal_ordinal": 1,
            "decision_source_event_sha256": row["decision_source_event_sha256"],
            "completed_data_watermark_source_ts_ns": row[
                "completed_data_watermark_source_ts_ns"
            ],
            "completed_data_prefix_root_sha256": row[
                "completed_data_prefix_root_sha256"
            ],
        }],
    }))
    ledger_bytes = _literal_ledger_bytes(fixture)
    ledger_rows = _literal_ledger_rows(fixture)
    metrics = _literal_metrics(fixture)
    journal_root = _literal_journal_root(fixture)
    projection = {
        "all_transactions_balanced": True,
        "engine_id": EXPECTED_ENGINE_ID,
        "input_root_sha256": input_root,
        "journal_root_sha256": journal_root,
        "journal_transaction_count": len(_literal_journal_transactions(fixture)),
        "ledger_row_count": len(ledger_rows),
        "ledger_sha256": _digest(ledger_bytes),
        "ledger_terminal_hash": ledger_rows[-1]["record_hash"],
        "oracle_metrics_sha256": metrics["metrics_sha256"],
        "proposal_provenance_root_sha256": proposal_root,
    }
    return {
        "input_root_sha256": input_root,
        "proposal_provenance_root_sha256": proposal_root,
        "journal_root_sha256": journal_root,
        "economic_projection_sha256": _digest(_canonical(projection)),
    }


def _ledger_rows(raw: bytes) -> list[dict[str, Any]]:
    assert type(raw) is bytes
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    rows = [json.loads(line) for line in raw.splitlines()]
    assert all(_canonical(row) + b"\n" in raw for row in rows)
    return rows


def _replace_json_artifact(
    artifacts: Mapping[str, bytes],
    label: str,
    hash_field: str,
    mutate: Callable[[dict[str, Any]], None],
) -> dict[str, bytes]:
    result = dict(artifacts)
    value = json.loads(result[label])
    value.pop(hash_field, None)
    mutate(value)
    result[label] = _json_artifact(_seal(value, hash_field))
    return result


def _future_price_artifacts(
    fixture: CanonicalFixture,
    delta_ticks: int,
) -> dict[str, bytes]:
    rows = [dict(row) for row in fixture.source_rows]
    rows[-1]["bid_ticks"] += delta_ticks
    rows[-1]["ask_ticks"] += delta_ticks
    source_blob = b"".join(_canonical(row) + b"\n" for row in rows)
    manifest = deepcopy(dict(fixture.values["source_manifest"]))
    manifest.pop("manifest_sha256")
    manifest["source_bytes_sha256"] = _digest(source_blob)
    manifest["source_size_bytes"] = len(source_blob)
    result = dict(fixture.artifacts)
    result["source_blob"] = source_blob
    result["source_manifest"] = _json_artifact(_seal(manifest, "manifest_sha256"))
    return result


def _replace_last_source_event(
    fixture: CanonicalFixture,
    *,
    source_ts_ns: int,
    arrival_ts_ns: int,
    bid_ticks: int,
    ask_ticks: int,
) -> dict[str, bytes]:
    rows = [dict(row) for row in fixture.source_rows]
    rows[-1].update({
        "source_ts_ns": source_ts_ns,
        "arrival_ts_ns": arrival_ts_ns,
        "bid_ticks": bid_ticks,
        "ask_ticks": ask_ticks,
    })
    source_blob = b"".join(_canonical(row) + b"\n" for row in rows)
    manifest = deepcopy(dict(fixture.values["source_manifest"]))
    manifest.pop("manifest_sha256")
    manifest["source_bytes_sha256"] = _digest(source_blob)
    manifest["source_size_bytes"] = len(source_blob)
    manifest["first_source_ts_ns"] = min(row["source_ts_ns"] for row in rows)
    manifest["last_source_ts_ns"] = max(row["source_ts_ns"] for row in rows)
    stream = manifest["stream_policies"][0]
    stream["max_source_gap_ns"] = max(
        later["source_ts_ns"] - earlier["source_ts_ns"]
        for earlier, later in zip(rows, rows[1:])
    )
    stream["max_arrival_gap_ns"] = max(
        later["arrival_ts_ns"] - earlier["arrival_ts_ns"]
        for earlier, later in zip(rows, rows[1:])
    )
    result = dict(fixture.artifacts)
    result["source_blob"] = source_blob
    result["source_manifest"] = _json_artifact(_seal(
        manifest,
        "manifest_sha256",
    ))
    return result


def _boundary_risk_artifacts(
    fixture: CanonicalFixture,
    close_clock_ns: int,
) -> dict[str, bytes]:
    # The last real quote remains at START + one day + one second.  The loss
    # that crosses the risk threshold is financing accrued after that quote,
    # so a close at ``close_clock_ns`` proves the reducer generated a
    # synthetic month/terminal checkpoint rather than merely reacting to a
    # market event placed at the boundary.
    artifacts = dict(fixture.artifacts)
    artifacts = _replace_json_artifact(
        artifacts,
        "proposal",
        "proposal_sha256",
        lambda payload: payload["rows"][0].__setitem__(
            "max_age_ns", MARCH_START_NS - START_NS + DAY_NS
        ),
    )

    def execution(payload: dict[str, Any]) -> None:
        payload["max_trade_quote_staleness_ns"] = MARCH_START_NS - START_NS
        payload["arms"]["EXECUTABLE_BASE"]["financing_ppm_per_day"] = 100_000
        payload["arms"]["ADVERSE_STRESS"]["financing_ppm_per_day"] = 100_000

    artifacts = _replace_json_artifact(
        artifacts,
        "execution_policy",
        "execution_policy_sha256",
        execution,
    )
    artifacts = _replace_json_artifact(
        artifacts,
        "accounting_policy",
        "accounting_policy_sha256",
        lambda payload: payload.__setitem__(
            "max_conversion_staleness_ns", MARCH_START_NS - START_NS
        ),
    )

    def evaluation(payload: dict[str, Any]) -> None:
        payload["period_end_ts_ns"] = MARCH_START_NS
        payload["initial_equity_jpy_micros"] = (
            200_000_000 if close_clock_ns == END_NS - 1 else 400_000_000
        )
        payload["margin_rate_bps"] = 500
        payload["max_gross_to_equity_bps"] = 100_000
        payload["full_month_ids"] = ["2026-01", "2026-02"]

    return _replace_json_artifact(
        artifacts,
        "evaluation_policy",
        "evaluation_policy_sha256",
        evaluation,
    )


def _short_artifacts(fixture: CanonicalFixture) -> dict[str, bytes]:
    def make_short(payload: dict[str, Any]) -> None:
        payload["candidate_key"] = "DOUBLE-ENTRY-GOLDEN-USDJPY-SHORT"
        payload["rows"][0]["direction"] = -1

    return _replace_json_artifact(
        fixture.artifacts,
        "proposal",
        "proposal_sha256",
        make_short,
    )


def _pip_artifacts(fixture: CanonicalFixture) -> dict[str, bytes]:
    artifacts = _replace_json_artifact(
        fixture.artifacts,
        "instrument_registry",
        "registry_sha256",
        lambda payload: payload["instruments"]["USD_JPY"].__setitem__(
            "pip_ticks", 5
        ),
    )
    registry = json.loads(artifacts["instrument_registry"])
    artifacts = _replace_json_artifact(
        artifacts,
        "source_manifest",
        "manifest_sha256",
        lambda payload: payload.__setitem__(
            "instrument_registry_sha256",
            registry["registry_sha256"],
        ),
    )
    return artifacts


def _terminal_artifacts(fixture: CanonicalFixture) -> dict[str, bytes]:
    artifacts = _replace_json_artifact(
        fixture.artifacts,
        "proposal",
        "proposal_sha256",
        lambda payload: payload["rows"][0].__setitem__(
            "max_age_ns", END_NS - START_NS + DAY_NS
        ),
    )
    artifacts = _replace_json_artifact(
        artifacts,
        "execution_policy",
        "execution_policy_sha256",
        lambda payload: payload.__setitem__(
            "max_trade_quote_staleness_ns", END_NS - START_NS
        ),
    )
    return _replace_json_artifact(
        artifacts,
        "accounting_policy",
        "accounting_policy_sha256",
        lambda payload: payload.__setitem__(
            "max_conversion_staleness_ns", END_NS - START_NS
        ),
    )


def _short_gross_cap_artifacts(fixture: CanonicalFixture) -> dict[str, bytes]:
    return _replace_json_artifact(
        _short_artifacts(fixture),
        "inventory_policy",
        "inventory_policy_sha256",
        lambda payload: payload.__setitem__(
            "max_gross_notional_jpy_micros", 100_010_000
        ),
    )


def _currency_cap_artifacts(fixture: CanonicalFixture) -> dict[str, bytes]:
    return _replace_json_artifact(
        fixture.artifacts,
        "inventory_policy",
        "inventory_policy_sha256",
        lambda payload: payload.__setitem__(
            "max_currency_notional_jpy_micros", 99_000_000
        ),
    )


def _margin_entry_artifacts(fixture: CanonicalFixture) -> dict[str, bytes]:
    return _replace_json_artifact(
        fixture.artifacts,
        "evaluation_policy",
        "evaluation_policy_sha256",
        lambda payload: payload.__setitem__(
            "initial_equity_jpy_micros", 5_000_000
        ),
    )


def _inverse_conversion_fixture(
    fixture: CanonicalFixture,
    currency: str,
) -> MatrixFixture:
    trade = f"EUR_{currency}"
    conversion = f"USD_{currency}"
    events = (
        {
            "tag": "usd-jpy",
            "instrument": "USD_JPY",
            "bid_ticks": 15_000,
            "ask_ticks": 15_002,
            "source_ts_ns": START_NS,
            "arrival_ts_ns": START_NS,
        },
        {
            "tag": "inverse",
            "instrument": conversion,
            "bid_ticks": 125,
            "ask_ticks": 126,
            "source_ts_ns": START_NS + 1_000_000_000,
            "arrival_ts_ns": START_NS + 1_000_000_000,
        },
        {
            "tag": "decision",
            "instrument": trade,
            "bid_ticks": 149,
            "ask_ticks": 151,
            "source_ts_ns": START_NS + 2_000_000_000,
            "arrival_ts_ns": START_NS + 2_000_000_000,
        },
        {
            "tag": "entry",
            "instrument": trade,
            "bid_ticks": 149,
            "ask_ticks": 151,
            "source_ts_ns": START_NS + 3_000_000_000,
            "arrival_ts_ns": START_NS + 3_000_000_000,
        },
        {
            "tag": "exit",
            "instrument": trade,
            "bid_ticks": 159,
            "ask_ticks": 161,
            "source_ts_ns": START_NS + 4_000_000_000,
            "arrival_ts_ns": START_NS + 4_000_000_000,
        },
    )
    return _build_matrix_fixture(
        fixture,
        events=events,
        proposals=({
            "decision_tag": "decision",
            "instrument": trade,
            "direction": 1,
            "max_age_ns": 1,
        },),
        registry={
            trade: {"price_scale": 100, "pip_ticks": 1},
            conversion: {"price_scale": 100, "pip_ticks": 1},
            "USD_JPY": {"price_scale": 100, "pip_ticks": 1},
        },
    )


def _usd_jpy_event(tag: str, seconds: int, mid_ticks: int) -> dict[str, Any]:
    return {
        "tag": tag,
        "instrument": "USD_JPY",
        "bid_ticks": mid_ticks - 1,
        "ask_ticks": mid_ticks + 1,
        "source_ts_ns": START_NS + seconds * 1_000_000_000,
        "arrival_ts_ns": START_NS + seconds * 1_000_000_000,
    }


def _collision_fixture(fixture: CanonicalFixture) -> MatrixFixture:
    return _build_matrix_fixture(
        fixture,
        events=(
            _usd_jpy_event("decision-1", 0, 10_000),
            _usd_jpy_event("entry-1", 1, 10_000),
            _usd_jpy_event("decision-2", 2, 10_000),
            _usd_jpy_event("entry-2", 3, 10_000),
            _usd_jpy_event("exit-2", 4, 10_050),
            _usd_jpy_event("exit-1", 11, 10_100),
        ),
        proposals=(
            {
                "decision_tag": "decision-1",
                "instrument": "USD_JPY",
                "direction": 1,
                "max_age_ns": 10_000_000_000,
            },
            {
                "decision_tag": "decision-2",
                "instrument": "USD_JPY",
                "direction": -1,
                "max_age_ns": 1_000_000_000,
            },
        ),
        registry={"USD_JPY": {"price_scale": 100, "pip_ticks": 1}},
    )


def _ruin_fixture(fixture: CanonicalFixture) -> MatrixFixture:
    def loosen_entry_ratio(payload: dict[str, Any]) -> None:
        payload["initial_equity_jpy_micros"] = 50_000_000
        payload["max_gross_to_equity_bps"] = 100_000

    return _build_matrix_fixture(
        fixture,
        events=(
            _usd_jpy_event("decision-1", 0, 10_000),
            _usd_jpy_event("entry-1", 1, 10_000),
            _usd_jpy_event("decision-2", 2, 10_000),
            _usd_jpy_event("crash-entry-2", 3, 1_000),
            _usd_jpy_event("common-exit-2", 4, 1_100),
            _usd_jpy_event("common-exit-1", 6, 1_200),
        ),
        proposals=(
            {
                "decision_tag": "decision-1",
                "instrument": "USD_JPY",
                "direction": 1,
                "max_age_ns": 5_000_000_000,
            },
            {
                "decision_tag": "decision-2",
                "instrument": "USD_JPY",
                "direction": 1,
                "notional_jpy_micros": 10_000_000,
                "max_age_ns": 1_000_000_000,
            },
        ),
        registry={"USD_JPY": {"price_scale": 100, "pip_ticks": 1}},
        mutate_evaluation=loosen_entry_ratio,
    )


def _three_cluster_fixture(fixture: CanonicalFixture) -> MatrixFixture:
    events: list[Mapping[str, Any]] = []
    proposals: list[Mapping[str, Any]] = []
    exit_mids = (9_900, 10_200, 9_700)
    for ordinal, (base_second, exit_mid) in enumerate(
        zip((0, 10, 20), exit_mids),
        1,
    ):
        events.extend((
            _usd_jpy_event(f"decision-{ordinal}", base_second, 10_000),
            _usd_jpy_event(f"entry-{ordinal}", base_second + 1, 10_000),
            _usd_jpy_event(f"exit-{ordinal}", base_second + 2, exit_mid),
        ))
        proposals.append({
            "decision_tag": f"decision-{ordinal}",
            "instrument": "USD_JPY",
            "direction": 1,
            "max_age_ns": 1_000_000_000,
        })
    return _build_matrix_fixture(
        fixture,
        events=tuple(events),
        proposals=tuple(proposals),
        registry={"USD_JPY": {"price_scale": 100, "pip_ticks": 1}},
        mutate_evaluation=lambda payload: payload.__setitem__(
            "cvar_tail_bps", 3_334
        ),
    )


def _safe_multi_terminal_fixture(fixture: CanonicalFixture) -> MatrixFixture:
    terminal_end = START_NS + 10_000_000_000

    def execution(payload: dict[str, Any]) -> None:
        payload["max_trade_quote_staleness_ns"] = 20_000_000_000

    def inventory(payload: dict[str, Any]) -> None:
        payload["max_open_positions"] = 2

    def accounting(payload: dict[str, Any]) -> None:
        payload["max_conversion_staleness_ns"] = 20_000_000_000

    def evaluation(payload: dict[str, Any]) -> None:
        payload["period_end_ts_ns"] = terminal_end
        payload["full_month_ids"] = []

    events = (
        _usd_jpy_event("usd-decision", 0, 10_000),
        {
            "tag": "chf-decision",
            "instrument": "CHF_JPY",
            "bid_ticks": 16_999,
            "ask_ticks": 17_001,
            "source_ts_ns": START_NS + 1_000_000_000,
            "arrival_ts_ns": START_NS + 1_000_000_000,
        },
        _usd_jpy_event("usd-entry", 2, 10_000),
        {
            "tag": "chf-entry",
            "instrument": "CHF_JPY",
            "bid_ticks": 16_999,
            "ask_ticks": 17_001,
            "source_ts_ns": START_NS + 3_000_000_000,
            "arrival_ts_ns": START_NS + 3_000_000_000,
        },
        _usd_jpy_event("usd-mark", 4, 10_050),
        {
            "tag": "chf-mark",
            "instrument": "CHF_JPY",
            "bid_ticks": 17_099,
            "ask_ticks": 17_101,
            "source_ts_ns": START_NS + 5_000_000_000,
            "arrival_ts_ns": START_NS + 5_000_000_000,
        },
    )
    return _build_matrix_fixture(
        fixture,
        events=events,
        proposals=(
            {
                "decision_tag": "usd-decision",
                "instrument": "USD_JPY",
                "direction": 1,
                "max_age_ns": 20_000_000_000,
            },
            {
                "decision_tag": "chf-decision",
                "instrument": "CHF_JPY",
                "direction": 1,
                "max_age_ns": 20_000_000_000,
            },
        ),
        registry={
            "CHF_JPY": {"price_scale": 100, "pip_ticks": 1},
            "USD_JPY": {"price_scale": 100, "pip_ticks": 1},
        },
        mutate_execution=execution,
        mutate_inventory=inventory,
        mutate_accounting=accounting,
        mutate_evaluation=evaluation,
    )


def _assert_sha256(value: Any) -> None:
    assert type(value) is str
    assert len(value) == 64
    assert set(value) <= set("0123456789abcdef")


def test_public_api_is_small_selector_free_and_typed(
    canonical_fixture: CanonicalFixture,
) -> None:
    assert reference.__all__ == (
        "ENGINE_ID",
        "ReferenceError",
        "decode_reference_input",
        "replay_reference",
    )
    assert reference.ENGINE_ID == EXPECTED_ENGINE_ID
    assert issubclass(reference.ReferenceError, RuntimeError)
    assert list(inspect.signature(reference.decode_reference_input).parameters) == [
        "artifacts"
    ]
    assert list(inspect.signature(reference.replay_reference).parameters) == [
        "artifacts"
    ]
    with pytest.raises(TypeError):
        reference.replay_reference(  # type: ignore[call-arg]
            canonical_fixture.artifacts,
            mutation="MK01A",
        )


def test_decode_accepts_a_general_mapping_and_binds_all_nine_raw_bytes(
    canonical_fixture: CanonicalFixture,
) -> None:
    reads: Counter[str] = Counter()

    class SingleReadMapping(Mapping[str, bytes]):
        def __iter__(self):
            return iter(canonical_fixture.artifacts)

        def __len__(self) -> int:
            return len(canonical_fixture.artifacts)

        def __getitem__(self, key: str) -> bytes:
            reads[key] += 1
            if reads[key] != 1:
                raise AssertionError(f"artifact reread: {key}")
            return canonical_fixture.artifacts[key]

    decoded = reference.decode_reference_input(SingleReadMapping())
    assert decoded.candidate_key == "DOUBLE-ENTRY-GOLDEN-USDJPY-LONG"
    assert len(decoded.ticks) == 3
    assert len(decoded.proposals) == 1
    assert set(decoded.books) == {"USD_JPY"}
    assert set(decoded.raw_hashes) == set(ARTIFACT_KEYS)
    assert decoded.raw_hashes == {
        key: _digest(canonical_fixture.artifacts[key])
        for key in sorted(ARTIFACT_KEYS)
    }
    assert reads == Counter({key: 1 for key in ARTIFACT_KEYS})
    with pytest.raises(TypeError):
        decoded.authority["paper_only"] = False  # type: ignore[index]
    with pytest.raises(TypeError):
        decoded.registry["USD_JPY"]["price_scale"] = 1  # type: ignore[index]
    assert decoded.authority == {
        "schema_version": 2,
        "policy_id": "FROZEN_PAPER_AUTHORITY_V1",
        "paper_only": True,
        "live_authority": False,
        "broker_account_access": False,
        "credential_access": False,
        "order_endpoint": False,
        "external_orders": 0,
        "deploy": False,
        "external_config_mutation": False,
        "authority_policy_sha256": canonical_fixture.values["authority_policy"][
            "authority_policy_sha256"
        ],
    }


def test_literal_replay_matches_exact_ledger_metrics_and_roots(
    canonical_fixture: CanonicalFixture,
) -> None:
    result = reference.replay_reference(dict(canonical_fixture.artifacts))
    literal_ledger = _literal_ledger_bytes(canonical_fixture)
    literal_rows = _literal_ledger_rows(canonical_fixture)
    literal_roots = _literal_result_roots(canonical_fixture)
    assert set(result) == set(EXPECTED_RESULT_KEYS)
    assert result["engine_id"] == EXPECTED_ENGINE_ID
    assert result["ledger_bytes"] == literal_ledger
    assert result["ledger_row_count"] == 3
    assert result["ledger_terminal_hash"] == literal_rows[-1]["record_hash"]
    assert result["oracle_metrics"] == _literal_metrics(canonical_fixture)
    assert result["all_transactions_balanced"] is True
    assert result["journal_transaction_count"] == len(
        _literal_journal_transactions(canonical_fixture)
    ) == 22
    for key, expected in literal_roots.items():
        assert result[key] == expected
        _assert_sha256(result[key])


def test_replay_is_pure_deterministic_and_mapping_order_independent(
    canonical_fixture: CanonicalFixture,
) -> None:
    original_bytes = dict(canonical_fixture.artifacts)
    reverse_order = dict(reversed(tuple(canonical_fixture.artifacts.items())))
    first = reference.replay_reference(dict(canonical_fixture.artifacts))
    second = reference.replay_reference(reverse_order)
    third = reference.replay_reference(dict(canonical_fixture.artifacts))
    assert first == second == third
    assert dict(canonical_fixture.artifacts) == original_bytes


@pytest.mark.parametrize("label", sorted(ARTIFACT_KEYS))
def test_exact_nine_artifact_boundary_rejects_missing_and_nonbytes(
    canonical_fixture: CanonicalFixture,
    label: str,
) -> None:
    missing = dict(canonical_fixture.artifacts)
    missing.pop(label)
    with pytest.raises(reference.ReferenceError):
        reference.decode_reference_input(missing)
    wrong_type: dict[str, Any] = dict(canonical_fixture.artifacts)
    wrong_type[label] = bytearray(wrong_type[label])
    with pytest.raises(reference.ReferenceError):
        reference.decode_reference_input(wrong_type)


def test_exact_nine_artifact_boundary_rejects_unknown_role(
    canonical_fixture: CanonicalFixture,
) -> None:
    artifacts = dict(canonical_fixture.artifacts)
    artifacts["oracle_ledger"] = b"producer output must never be an input\n"
    with pytest.raises(reference.ReferenceError):
        reference.decode_reference_input(artifacts)


def test_strict_decoder_rejects_noncanonical_duplicate_and_negative_zero(
    canonical_fixture: CanonicalFixture,
) -> None:
    for raw in (
        b'{"registry_id":"x", "schema_version":1}\n',
        b'{"registry_id":"x","registry_id":"y"}\n',
        b'{"schema_version":-0}\n',
        b'{"schema_version":1.0}\n',
        b'{"schema_version":NaN}\n',
    ):
        artifacts = dict(canonical_fixture.artifacts)
        artifacts["instrument_registry"] = raw
        with pytest.raises(reference.ReferenceError):
            reference.decode_reference_input(artifacts)


def test_manifest_byte_binding_and_proposal_causal_binding_fail_closed(
    canonical_fixture: CanonicalFixture,
) -> None:
    corrupted_blob = dict(canonical_fixture.artifacts)
    corrupted_blob["source_blob"] = corrupted_blob["source_blob"].replace(
        b'"bid_ticks":10099', b'"bid_ticks":10098', 1
    )
    with pytest.raises(reference.ReferenceError):
        reference.decode_reference_input(corrupted_blob)

    bad_prefix = _replace_json_artifact(
        canonical_fixture.artifacts,
        "proposal",
        "proposal_sha256",
        lambda value: value["rows"][0].__setitem__(
            "completed_data_prefix_root_sha256", "f" * 64
        ),
    )
    with pytest.raises(reference.ReferenceError):
        reference.decode_reference_input(bad_prefix)


def test_producer_outcome_fields_are_recursively_rejected_even_when_resealed(
    canonical_fixture: CanonicalFixture,
) -> None:
    for forbidden_key in (
        "PnL",
        "realized_pnl_jpy_micros",
        "fill_price",
        "ending_equity_jpy_micros",
        "max_drawdown_ratio",
        "profit_gate",
    ):
        artifacts = _replace_json_artifact(
            canonical_fixture.artifacts,
            "proposal",
            "proposal_sha256",
            lambda value, key=forbidden_key: value["rows"][0].__setitem__(
                "nested", {key: 1}
            ),
        )
        with pytest.raises(reference.ReferenceError):
            reference.decode_reference_input(artifacts)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("paper_only", False),
        ("live_authority", True),
        ("broker_account_access", True),
        ("credential_access", True),
        ("order_endpoint", True),
        ("external_orders", 1),
        ("external_orders", False),
        ("deploy", True),
        ("external_config_mutation", True),
    ),
)
def test_authority_is_exact_typed_and_fail_closed_when_resealed(
    canonical_fixture: CanonicalFixture,
    field: str,
    value: Any,
) -> None:
    artifacts = _replace_json_artifact(
        canonical_fixture.artifacts,
        "authority_policy",
        "authority_policy_sha256",
        lambda payload: payload.__setitem__(field, value),
    )
    with pytest.raises(reference.ReferenceError):
        reference.decode_reference_input(artifacts)


def test_authority_cannot_be_redefined_through_mutable_module_state(
    canonical_fixture: CanonicalFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = _replace_json_artifact(
        canonical_fixture.artifacts,
        "authority_policy",
        "authority_policy_sha256",
        lambda payload: payload.__setitem__("live_authority", True),
    )
    monkeypatch.setattr(
        reference,
        "AUTHORITY",
        {"paper_only": True, "live_authority": True},
        raising=False,
    )
    with pytest.raises(reference.ReferenceError):
        reference.replay_reference(live)


def test_first_replay_after_import_performs_no_lazy_import(
    canonical_fixture: CanonicalFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted: list[str] = []

    def deny_import(name: str, *args: Any, **kwargs: Any) -> Any:
        attempted.append(name)
        raise AssertionError(f"late import blocked: {name}")

    monkeypatch.setattr(builtins, "__import__", deny_import)
    result = reference.replay_reference(dict(canonical_fixture.artifacts))
    assert result["all_transactions_balanced"] is True
    assert attempted == []


def test_future_price_perturbation_preserves_prior_signal_identity_only(
    canonical_fixture: CanonicalFixture,
) -> None:
    baseline = reference.replay_reference(dict(canonical_fixture.artifacts))
    perturbed = reference.replay_reference(
        _future_price_artifacts(canonical_fixture, 25)
    )
    baseline_rows = _ledger_rows(baseline["ledger_bytes"])
    perturbed_rows = _ledger_rows(perturbed["ledger_bytes"])
    assert {row["signal_id"] for row in baseline_rows} == {
        row["signal_id"] for row in perturbed_rows
    }
    assert {row["economic_lot_id"] for row in baseline_rows} == {
        row["economic_lot_id"] for row in perturbed_rows
    }
    assert baseline["proposal_provenance_root_sha256"] == perturbed[
        "proposal_provenance_root_sha256"
    ]
    assert baseline["input_root_sha256"] != perturbed["input_root_sha256"]
    assert baseline["ledger_bytes"] != perturbed["ledger_bytes"]
    assert baseline_rows[0]["entry_source_event_sha256"] == perturbed_rows[0][
        "entry_source_event_sha256"
    ]
    assert baseline_rows[0]["exit_source_event_sha256"] != perturbed_rows[0][
        "exit_source_event_sha256"
    ]


def test_tightening_only_adverse_commission_has_monotone_economic_effect(
    canonical_fixture: CanonicalFixture,
) -> None:
    baseline = reference.replay_reference(dict(canonical_fixture.artifacts))

    def tighten(payload: dict[str, Any]) -> None:
        payload["arms"]["ADVERSE_STRESS"]["commission_ppm_per_side"] = 30

    tightened = reference.replay_reference(_replace_json_artifact(
        canonical_fixture.artifacts,
        "execution_policy",
        "execution_policy_sha256",
        tighten,
    ))
    for arm in ("RAW_SIGNAL", "EXECUTABLE_BASE"):
        assert baseline["oracle_metrics"]["arms"][arm] == tightened[
            "oracle_metrics"
        ]["arms"][arm]
    assert tightened["oracle_metrics"]["arms"]["ADVERSE_STRESS"][
        "realized_cost_jpy_micros"
    ] > baseline["oracle_metrics"]["arms"]["ADVERSE_STRESS"][
        "realized_cost_jpy_micros"
    ]
    assert tightened["oracle_metrics"]["arms"]["ADVERSE_STRESS"][
        "net_pnl_jpy_micros"
    ] < baseline["oracle_metrics"]["arms"]["ADVERSE_STRESS"][
        "net_pnl_jpy_micros"
    ]


def test_short_uses_bid_to_open_and_ask_to_close_with_negative_flooring(
    canonical_fixture: CanonicalFixture,
) -> None:
    rows = _ledger_rows(reference.replay_reference(
        _short_artifacts(canonical_fixture)
    )["ledger_bytes"])
    executable = next(row for row in rows if row["arm"] == "EXECUTABLE_BASE")
    assert executable["direction"] == -1
    assert (
        executable["entry_price_numerator"],
        executable["entry_price_denominator"],
    ) == (9_999_000_000, 100_000_000)
    assert (
        executable["exit_price_numerator"],
        executable["exit_price_denominator"],
    ) == (10_101_000_000, 100_000_000)
    assert executable["units_micros"] == 1_000_100
    assert executable["executable_pnl_before_direct_cost_jpy_micros"] == -1_020_102
    assert executable["net_pnl_jpy_micros"] == -1_022_613


@pytest.mark.parametrize("currency", ("CAD", "CHF"))
def test_inverse_quote_conversion_divides_then_converts_usd_to_jpy(
    canonical_fixture: CanonicalFixture,
    currency: str,
) -> None:
    fixture = _inverse_conversion_fixture(canonical_fixture, currency)
    rows = _ledger_rows(reference.replay_reference(fixture.artifacts)["ledger_bytes"])
    row = next(item for item in rows if item["arm"] == "EXECUTABLE_BASE")
    entry_price = Fraction(151, 100)
    liability_jpy_per_unit = entry_price / Fraction(125, 100) * Fraction(15_002, 100)
    per_unit_micros = liability_jpy_per_unit * 1_000_000
    expected_units = int(Fraction(100_000_000 * 1_000_000, 1) // per_unit_micros)
    positive_quote_pnl = (
        Fraction(expected_units, 1_000_000)
        * (Fraction(159, 100) - entry_price)
    )
    executable_exact = (
        positive_quote_pnl
        / Fraction(126, 100)
        * Fraction(15_000, 100)
        * 1_000_000
    )
    assert row["entry_price_numerator"] == 151_000_000
    assert row["entry_price_denominator"] == 100_000_000
    assert row["exit_price_numerator"] == 159_000_000
    assert row["exit_price_denominator"] == 100_000_000
    assert row["units_micros"] == expected_units
    assert row["executable_pnl_before_direct_cost_jpy_micros"] == (
        executable_exact.numerator // executable_exact.denominator
    )


def test_opposite_direction_same_pair_collides_without_netting(
    canonical_fixture: CanonicalFixture,
) -> None:
    rows = _ledger_rows(reference.replay_reference(
        _collision_fixture(canonical_fixture).artifacts
    )["ledger_bytes"])
    by_arm_ordinal = {
        (row["arm"], row["proposal_ordinal"]): row
        for row in rows
    }
    for arm in ARMS:
        assert by_arm_ordinal[(arm, 1)]["status"] == "FILLED_CLOSED"
        rejected = by_arm_ordinal[(arm, 2)]
        assert rejected["direction"] == -1
        assert rejected["status"] == "SAME_PAIR_COLLISION_REJECTED"
        assert rejected["units_micros"] == 0


def test_collision_attribution_is_settled_only_at_common_exit_clock(
    canonical_fixture: CanonicalFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, int, int, str]] = []
    original_post = reference._Journal.post

    def tracing_post(self: Any, **kwargs: Any) -> None:
        before = len(self.transactions)
        original_post(self, **kwargs)
        if len(self.transactions) != before:
            event = self.transactions[-1]
            observed.append((
                event.arm,
                event.proposal_ordinal,
                event.arrival_ts_ns,
                event.event_kind,
            ))

    monkeypatch.setattr(reference._Journal, "post", tracing_post)
    reference.replay_reference(_collision_fixture(canonical_fixture).artifacts)
    rejection_clock = START_NS + 3_000_000_000
    common_exit_clock = START_NS + 4_000_000_000
    for arm in ARMS:
        transactions = [
            item for item in observed if item[0] == arm and item[1] == 2
        ]
        assert all(item[2] >= rejection_clock for item in transactions)
        disposition = next(
            item for item in transactions if item[3] == "PROPOSAL_DISPOSITION"
        )
        assert disposition[2] == common_exit_clock
        assert [item[2] for item in observed if item[0] == arm] == sorted(
            item[2] for item in observed if item[0] == arm
        )


def test_margin_ruin_closes_inventory_then_halts_later_admission(
    canonical_fixture: CanonicalFixture,
) -> None:
    result = reference.replay_reference(_ruin_fixture(canonical_fixture).artifacts)
    rows = _ledger_rows(result["ledger_bytes"])
    by_arm_ordinal = {
        (row["arm"], row["proposal_ordinal"]): row
        for row in rows
    }
    for arm in ARMS:
        assert by_arm_ordinal[(arm, 1)]["exit_disposition"] == "MARGIN_CLOSEOUT"
        assert by_arm_ordinal[(arm, 2)]["status"] == "ACCOUNT_HALTED"
        metrics = result["oracle_metrics"]["arms"][arm]
        assert metrics["margin_guard_pass"] is False
        assert metrics["terminal_open_positions"] == 0


def test_safe_terminal_freezes_and_marks_all_positions_before_any_close(
    canonical_fixture: CanonicalFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal_clock = START_NS + 10_000_000_000 - 1
    calls: list[tuple[str, int, int, str]] = []
    original_mark = reference._journal_mark
    original_settle = reference._settle_position

    def tracing_mark(
        data: Any,
        journal: Any,
        position: Any,
        tick: Any,
        arrival_ns: int,
    ) -> None:
        calls.append((position.arm, position.proposal.ordinal, arrival_ns, "MARK"))
        original_mark(data, journal, position, tick, arrival_ns)

    def tracing_settle(
        data: Any,
        journal: Any,
        position: Any,
        exit_tick: Any,
        exit_reason: str,
        **kwargs: Any,
    ) -> Any:
        arrival_ns = kwargs.get("valuation_arrival_ns", exit_tick.arrival_ts_ns)
        calls.append((position.arm, position.proposal.ordinal, arrival_ns, "CLOSE"))
        return original_settle(
            data,
            journal,
            position,
            exit_tick,
            exit_reason,
            **kwargs,
        )

    monkeypatch.setattr(reference, "_journal_mark", tracing_mark)
    monkeypatch.setattr(reference, "_settle_position", tracing_settle)
    result = reference.replay_reference(
        _safe_multi_terminal_fixture(canonical_fixture).artifacts
    )
    rows = _ledger_rows(result["ledger_bytes"])
    assert {row["exit_disposition"] for row in rows} == {"TERMINAL_LIQUIDATION"}
    for arm in ARMS:
        terminal_calls = [
            (ordinal, kind)
            for event_arm, ordinal, arrival_ns, kind in calls
            if event_arm == arm and arrival_ns == terminal_clock
        ]
        assert terminal_calls == [
            (1, "MARK"),
            (2, "MARK"),
            (1, "CLOSE"),
            (2, "CLOSE"),
        ]


def test_three_cluster_cvar_uses_ceiling_tail_count_and_two_worst_clusters(
    canonical_fixture: CanonicalFixture,
) -> None:
    result = reference.replay_reference(
        _three_cluster_fixture(canonical_fixture).artifacts
    )
    raw = result["oracle_metrics"]["arms"]["RAW_SIGNAL"]
    assert raw["currency_time_cluster_n_eff"] == 3
    assert sorted(
        row["cluster_risk_net_pnl_jpy_micros"]
        for row in raw["currency_time_cluster_observations"]
    ) == [-3_000_000, -1_000_000, 2_000_000]
    assert raw["cluster_cvar_jpy_micros"] == -2_000_000
    assert raw["cluster_cvar_return"] == "-0.002000000000000000"
    assert raw["ending_equity_jpy_micros"] == 998_000_000
    assert raw["max_drawdown_jpy_micros"] == 3_000_000
    assert raw["max_drawdown_ratio"] == "0.002997002997002998"


def test_gross_currency_and_margin_admission_guards_use_marked_values(
    canonical_fixture: CanonicalFixture,
) -> None:
    gross_rows = _ledger_rows(reference.replay_reference(
        _short_gross_cap_artifacts(canonical_fixture)
    )["ledger_bytes"])
    assert next(
        row for row in gross_rows if row["arm"] == "EXECUTABLE_BASE"
    )["status"] == "GROSS_CAP_REJECTED"

    currency_rows = _ledger_rows(reference.replay_reference(
        _currency_cap_artifacts(canonical_fixture)
    )["ledger_bytes"])
    assert next(
        row for row in currency_rows if row["arm"] == "EXECUTABLE_BASE"
    )["status"] == "CURRENCY_CAP_REJECTED"

    margin_rows = _ledger_rows(reference.replay_reference(
        _margin_entry_artifacts(canonical_fixture)
    )["ledger_bytes"])
    assert {row["status"] for row in margin_rows} == {"MARGIN_ENTRY_REJECTED"}


def test_empty_arm_months_remain_flat_and_no_fill_settles_at_terminal(
    canonical_fixture: CanonicalFixture,
) -> None:
    artifacts = _boundary_risk_artifacts(canonical_fixture, MARCH_START_NS - 1)

    def no_executable_fill(payload: dict[str, Any]) -> None:
        latency = MARCH_START_NS - START_NS
        payload["arms"]["EXECUTABLE_BASE"]["latency_ns"] = latency
        payload["arms"]["ADVERSE_STRESS"]["latency_ns"] = latency

    artifacts = _replace_json_artifact(
        artifacts,
        "execution_policy",
        "execution_policy_sha256",
        no_executable_fill,
    )
    result = reference.replay_reference(artifacts)
    rows = _ledger_rows(result["ledger_bytes"])
    for arm in ("EXECUTABLE_BASE", "ADVERSE_STRESS"):
        row = next(item for item in rows if item["arm"] == arm)
        assert row["status"] == "NO_CAUSAL_FILL"
        monthly = result["oracle_metrics"]["arms"][arm]["monthly"]
        assert len(monthly) == 2
        assert all(
            month["start_equity_jpy_micros"]
            == month["end_equity_jpy_micros"]
            for month in monthly
        )
        assert monthly[0]["end_equity_jpy_micros"] == monthly[1][
            "start_equity_jpy_micros"
        ]


def test_adverse_arm_orders_spread_pip_slippage_commission_and_financing(
    canonical_fixture: CanonicalFixture,
) -> None:
    rows = _ledger_rows(reference.replay_reference(
        _pip_artifacts(canonical_fixture)
    )["ledger_bytes"])
    by_arm = {row["arm"]: row for row in rows}
    entry_prices = {
        arm: Fraction(
            row["entry_price_numerator"], row["entry_price_denominator"]
        )
        for arm, row in by_arm.items()
    }
    exit_prices = {
        arm: Fraction(
            row["exit_price_numerator"], row["exit_price_denominator"]
        )
        for arm, row in by_arm.items()
    }
    assert entry_prices["RAW_SIGNAL"] < entry_prices["EXECUTABLE_BASE"] \
        < entry_prices["ADVERSE_STRESS"]
    assert exit_prices["RAW_SIGNAL"] > exit_prices["EXECUTABLE_BASE"] \
        > exit_prices["ADVERSE_STRESS"]
    assert by_arm["RAW_SIGNAL"]["commission_jpy_micros"] \
        < by_arm["EXECUTABLE_BASE"]["commission_jpy_micros"] \
        < by_arm["ADVERSE_STRESS"]["commission_jpy_micros"]
    assert by_arm["RAW_SIGNAL"]["financing_jpy_micros"] \
        < by_arm["EXECUTABLE_BASE"]["financing_jpy_micros"] \
        < by_arm["ADVERSE_STRESS"]["financing_jpy_micros"]
    assert by_arm["RAW_SIGNAL"]["net_pnl_jpy_micros"] \
        > by_arm["EXECUTABLE_BASE"]["net_pnl_jpy_micros"] \
        > by_arm["ADVERSE_STRESS"]["net_pnl_jpy_micros"]


def test_rejected_dispositions_keep_the_independent_economic_lot_identity(
    canonical_fixture: CanonicalFixture,
) -> None:
    def defer_executable_arms(payload: dict[str, Any]) -> None:
        latency = END_NS - START_NS
        payload["arms"]["EXECUTABLE_BASE"]["latency_ns"] = latency
        payload["arms"]["ADVERSE_STRESS"]["latency_ns"] = latency

    result = reference.replay_reference(_replace_json_artifact(
        canonical_fixture.artifacts,
        "execution_policy",
        "execution_policy_sha256",
        defer_executable_arms,
    ))
    rows = _ledger_rows(result["ledger_bytes"])
    assert [row["status"] for row in rows] == [
        "FILLED_CLOSED",
        "NO_CAUSAL_FILL",
        "NO_CAUSAL_FILL",
    ]
    assert len({row["signal_id"] for row in rows}) == 1
    assert len({row["economic_lot_id"] for row in rows}) == 1
    assert rows[0]["economic_lot_id"] != rows[0]["signal_id"]


def test_no_fill_is_terminally_known_and_journal_clocks_never_reverse(
    canonical_fixture: CanonicalFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def defer_executable_arms(payload: dict[str, Any]) -> None:
        latency = END_NS - START_NS
        payload["arms"]["EXECUTABLE_BASE"]["latency_ns"] = latency
        payload["arms"]["ADVERSE_STRESS"]["latency_ns"] = latency

    artifacts = _replace_json_artifact(
        canonical_fixture.artifacts,
        "execution_policy",
        "execution_policy_sha256",
        defer_executable_arms,
    )
    observed: list[tuple[str, int, str]] = []
    original_post = reference._Journal.post

    def tracing_post(self: Any, **kwargs: Any) -> None:
        count_before = len(self.transactions)
        original_post(self, **kwargs)
        if len(self.transactions) != count_before:
            transaction = self.transactions[-1]
            observed.append((
                transaction.arm,
                transaction.arrival_ts_ns,
                transaction.event_id,
            ))

    monkeypatch.setattr(reference._Journal, "post", tracing_post)
    reference.replay_reference(artifacts)
    for arm in ARMS:
        clocks = [clock for row_arm, clock, _ in observed if row_arm == arm]
        assert clocks == sorted(clocks)
    no_fill = [
        (arm, clock)
        for arm, clock, event_id in observed
        if event_id.endswith("REJECT:NO_CAUSAL_FILL")
    ]
    assert no_fill == [
        ("EXECUTABLE_BASE", END_NS - 1),
        ("ADVERSE_STRESS", END_NS - 1),
    ]


def test_journal_rejects_an_arm_local_clock_reversal() -> None:
    journal = reference._Journal()
    common = {
        "arrival_ts_ns": 2,
        "arm": "RAW_SIGNAL",
        "proposal_ordinal": 1,
        "event_kind": "POSITION_OPEN",
        "event_id": "later",
        "source_event_sha256": "1" * 64,
        "postings": (("POSITION_BASIS", Fraction(1)), ("POSITION_CONTROL", Fraction(-1))),
    }
    journal.post(**common)
    with pytest.raises(reference.ReferenceError, match="clock reversal"):
        journal.post(**{
            **common,
            "arrival_ts_ns": 1,
            "event_id": "earlier",
        })


@pytest.mark.parametrize(
    "expected_close_clock_ns",
    (
        END_NS - 1,
        MARCH_START_NS - 1,
    ),
    ids=("interior-month-boundary", "terminal-preliquidation"),
)
def test_boundary_risk_state_closes_before_reporting_margin_pass(
    canonical_fixture: CanonicalFixture,
    expected_close_clock_ns: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reducer_calls: list[tuple[str, int, str]] = []
    original_mark = reference._journal_mark
    original_settle = reference._settle_position

    def tracing_mark(
        data: Any,
        journal: Any,
        position: Any,
        tick: Any,
        arrival_ns: int,
    ) -> None:
        reducer_calls.append((position.arm, arrival_ns, "MARK"))
        original_mark(data, journal, position, tick, arrival_ns)

    def tracing_settle(
        data: Any,
        journal: Any,
        position: Any,
        exit_tick: Any,
        exit_reason: str,
        **kwargs: Any,
    ) -> Any:
        arrival_ns = kwargs.get("valuation_arrival_ns", exit_tick.arrival_ts_ns)
        reducer_calls.append((position.arm, arrival_ns, "CLOSE"))
        return original_settle(
            data,
            journal,
            position,
            exit_tick,
            exit_reason,
            **kwargs,
        )

    monkeypatch.setattr(reference, "_journal_mark", tracing_mark)
    monkeypatch.setattr(reference, "_settle_position", tracing_settle)
    result = reference.replay_reference(_boundary_risk_artifacts(
        canonical_fixture,
        expected_close_clock_ns,
    ))
    rows = _ledger_rows(result["ledger_bytes"])
    for row in rows:
        if row["arm"] == "RAW_SIGNAL":
            assert row["exit_disposition"] == "TERMINAL_LIQUIDATION"
            continue
        assert row["exit_disposition"] == "MARGIN_CLOSEOUT"
        assert row["exit_arrival_ts_ns"] == expected_close_clock_ns
        assert row["exit_source_reference"]["arrival_ts_ns"] == (
            START_NS + DAY_NS + 1_000_000_000
        )
        assert row["exit_source_reference"]["arrival_ts_ns"] < row[
            "exit_arrival_ts_ns"
        ]
    for arm in ("EXECUTABLE_BASE", "ADVERSE_STRESS"):
        metrics = result["oracle_metrics"]["arms"][arm]
        assert metrics["margin_guard_pass"] is False
        assert metrics["minimum_free_margin_jpy_micros"] < 0
        assert metrics["maximum_required_margin_jpy_micros"] > 0
        assert metrics["terminal_open_positions"] == 0
        assert [month["month_id"] for month in metrics["monthly"]] == [
            "2026-01",
            "2026-02",
        ]
        february = metrics["monthly"][1]
        if expected_close_clock_ns == END_NS - 1:
            assert february["start_equity_jpy_micros"] == february[
                "end_equity_jpy_micros"
            ]
        else:
            assert february["start_equity_jpy_micros"] > february[
                "end_equity_jpy_micros"
            ]
        terminal_events = [
            event_kind
            for event_arm, arrival_ns, event_kind in reducer_calls
            if event_arm == arm and arrival_ns == expected_close_clock_ns
        ]
        assert terminal_events == ["MARK", "CLOSE"]


def test_reference_import_graph_and_capability_surface_are_pure() -> None:
    source_path = Path(inspect.getsourcefile(reference) or "")
    assert source_path.name == "paper_research_double_entry_reference_v2.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    allowed_import_roots = {
        "__future__",
        "collections",
        "dataclasses",
        "fractions",
        "hashlib",
        "json",
        "re",
        "types",
        "typing",
    }
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert node.module is not None
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots <= allowed_import_roots
    assert "datetime" not in imported_roots
    assert all(
        not (
            isinstance(node, ast.Attribute)
            and node.attr in {"strptime", "fromisoformat"}
        )
        for node in ast.walk(tree)
    )

    forbidden_calls = {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "input",
        "open",
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called_names.isdisjoint(forbidden_calls)
    forbidden_attributes = {
        "connect",
        "fork",
        "open",
        "popen",
        "read_bytes",
        "request",
        "run",
        "send",
        "socket",
        "system",
        "unlink",
        "urlopen",
        "write",
        "write_bytes",
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called_attributes.isdisjoint(forbidden_attributes)

    forbidden_project_import_fragments = {
        "jpy_accounting",
        "oracle_v2",
        "oracle_verifier",
        "orchestrator",
        "result_validator",
        "runner",
        "shadow_jpy",
        "system_v3",
    }
    assert all(
        all(fragment not in root for fragment in forbidden_project_import_fragments)
        for root in imported_roots
    )


def test_reducer_commits_domain_events_before_any_ledger_projection(
    canonical_fixture: CanonicalFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reducer_source = inspect.getsource(reference._reduce_arm_events)
    assert "_project_closed_disposition" not in reducer_source
    assert "_project_rejected_disposition" not in reducer_source
    assert '"record_type"' not in reducer_source
    assert "_simulate_arm" not in Path(
        inspect.getsourcefile(reference) or ""
    ).read_text(encoding="utf-8")

    order: list[str] = []
    original_commit = reference._Journal.commit_disposition
    original_closed_projection = reference._project_closed_disposition
    original_rejected_projection = reference._project_rejected_disposition

    def tracing_commit(self: Any, *args: Any, **kwargs: Any) -> None:
        original_commit(self, *args, **kwargs)
        order.append("COMMIT")

    def tracing_closed_projection(*args: Any, **kwargs: Any) -> dict[str, Any]:
        order.append("PROJECT")
        return original_closed_projection(*args, **kwargs)

    def tracing_rejected_projection(*args: Any, **kwargs: Any) -> dict[str, Any]:
        order.append("PROJECT")
        return original_rejected_projection(*args, **kwargs)

    monkeypatch.setattr(reference._Journal, "commit_disposition", tracing_commit)
    monkeypatch.setattr(
        reference,
        "_project_closed_disposition",
        tracing_closed_projection,
    )
    monkeypatch.setattr(
        reference,
        "_project_rejected_disposition",
        tracing_rejected_projection,
    )
    reference.replay_reference(dict(canonical_fixture.artifacts))
    assert order.count("COMMIT") == 3
    assert order.count("PROJECT") == 3
    assert order == ["COMMIT"] * 3 + ["PROJECT"] * 3


def test_contract_reports_integrated_local_only_scope_without_overclaim() -> None:
    contract_path = Path(__file__).with_name(
        "PAPER_RESEARCH_DOUBLE_ENTRY_REFERENCE_CONTRACT_V2.json"
    )
    raw = contract_path.read_bytes()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    contract = json.loads(raw)
    assert contract["classification"] == (
        "FUTURE_ONLY_ACCOUNTING_ONLY_LOCAL_UNANCHORED_NOT_ADMISSIBLE"
    )
    assert contract["scope"][
        "historical_phase_1_reference_pair_implemented"
    ] is True
    assert contract["scope"][
        "verifier_consumes_immutable_canonical_reference_result"
    ] is True
    assert contract["scope"]["verifier_integration_complete"] is True
    assert contract["scope"]["profit_gate_wired_to_reference_engine"] is False
    phase = contract["historical_phase_1_test_status"]
    assert phase["test_local_audit_projection_mutants_killed"] == 36
    assert phase["test_local_audit_projection_survivors"] == 0
    runtime = phase["source_or_runtime_engine_mutation_campaign"]
    assert runtime == {
        "tested_engine_source_sha256": _digest(
            Path(__file__).with_name(
                "paper_research_double_entry_reference_v2.py"
            ).read_bytes()
        ),
        "runtime_mutation_harness_sha256": _digest(
            Path(__file__).with_name(
                "test_paper_research_double_entry_reference_v2_runtime_mutations.py"
            ).read_bytes()
        ),
        "operator_count": 36,
        "public_replay_operator_count": 34,
        "component_guard_operator_count": 2,
        "component_guard_operator_ids": ["MK03C", "MK11B"],
        "mutants_killed": 36,
        "survivors": 0,
        "harness_errors": 0,
        "fresh_in_memory_module_per_mutant": True,
        "exact_AST_site_cardinality_enforced": True,
        "mutation_probe_reachability_enforced": True,
        "component_guard_scope": (
            "ONLY_INVARIANTS_UNREACHABLE_FROM_SCHEMA_VALID_PUBLIC_REPLAY_"
            "WITH_EXPLICIT_REACHABILITY_REASON"
        ),
    }
    assert phase["source_or_runtime_engine_mutation_campaign_completed"] is True
    assert phase["full_eight_fixture_mutation_contract_completed"] is True
    assert phase["phase_1_projection_kills_count_as_full_engine_mutation_score"] is False
    assert (
        "MUTANT_REACHES_INDEPENDENT_ECONOMIC_REPLAY_OR_A_DECLARED_COMPONENT_"
        "GUARD_FOR_AN_INVARIANT_UNREACHABLE_FROM_SCHEMA_VALID_PUBLIC_REPLAY"
    ) in contract["mutation_kill_contract"]["kill_counts_only_when"]
    acceptance = contract["acceptance"]
    assert acceptance["historical_phase_1_additive_reference_pair_complete"] is True
    assert acceptance["all_36_named_engine_implementation_mutations_killed"] is True
    assert acceptance["historical_phase_1_independent_rereview_passed"] is True
    assert acceptance["phase_2_verifier_and_launcher_integration_complete"] is True
    assert acceptance[
        "json_contract_is_supplemental_and_content_pinned_by_launcher"
    ] is True
    provenance = contract["verification_and_provenance"]
    for field in (
        "reference_code_and_contract_are_launcher_owned_read_only_release_fds",
        "reference_code_sha256_pinned",
        "reference_contract_sha256_pinned",
        "verifier_receives_only_immutable_canonical_reference_result_bytes_and_trusted_hashes",
        "phase_2_release_fd_hash_pinning_complete",
        "reference_n_eff_and_direction_accuracy_are_accounting_diagnostics_only",
        "future_profit_gate_upgrade_requires_separate_future_oracle_metrics_contract",
    ):
        assert provenance[field] is True
    for field in (
        "statistical_admission_or_profit_gate_may_consume_reference_diagnostics",
        "execution_attestation_claim",
        "formal_correctness_claim",
        "external_anchor_claim",
        "release_evidence_eligible",
        "causal_signal_admission",
        "admission_eligible",
        "producer_metrics_used",
    ):
        assert provenance[field] is False
    assert provenance["content_binding_claim"] == (
        "PINNED_REFERENCE_CODE_AND_CONTRACT_PLUS_EXACT_NINE_INPUT_BYTES_AND_"
        "CANONICAL_REFERENCE_RESULT_HASH_LOCAL_ONLY"
    )
    capability = contract["capability_boundary"]
    assert capability["launcher_runtime_builtin_set_is_exact_and_minimal"] is True
    assert capability[
        "initialization_import_and_class_build_capabilities_removed_before_replay"
    ] is True
    assert capability[
        "reference_replay_audit_phase_denies_non_input_observation_capabilities"
    ] is True
    assert capability[
        "verifier_receives_live_reference_callable_module_or_namespace"
    ] is False


@dataclass
class AuditProjection:
    result: dict[str, Any]
    ledger_rows: list[dict[str, Any]]
    metrics: dict[str, Any]
    journal: list[dict[str, Any]]
    semantic_guards: dict[str, Any]


SEMANTIC_GUARDS = {
    "short_entry_side": "BID",
    "short_exit_side": "ASK",
    "inverse_conversion_operation": "DIVIDE",
    "future_conversion_quote_rejected": True,
    "stale_conversion_quote_rejected": True,
    "financing_clock": "ARRIVAL",
    "terminal_financing_clock": "PERIOD_CUTOFF",
    "entry_processed_after_due_exits": True,
    "opposite_same_pair_policy": "REJECT_NEW",
    "admissions_halt_after_ruin": True,
    "terminal_liquidation_required": True,
    "economic_lot_rounding": "AFTER_CLUSTER_AGGREGATION",
    "cluster_connectivity": "SHARED_CURRENCY_CONNECTED_COMPONENT",
    "cvar_tail_count_rounding": "CEILING",
}


def _audit_projection(
    fixture: CanonicalFixture,
    result: Mapping[str, Any],
) -> AuditProjection:
    return AuditProjection(
        result=deepcopy(dict(result)),
        ledger_rows=deepcopy(_ledger_rows(result["ledger_bytes"])),
        metrics=deepcopy(result["oracle_metrics"]),
        journal=deepcopy(_literal_journal_transactions(fixture)),
        semantic_guards=deepcopy(SEMANTIC_GUARDS),
    )


def _assert_literal_audit(
    fixture: CanonicalFixture,
    projection: AuditProjection,
) -> None:
    assert projection.ledger_rows == _literal_ledger_rows(fixture)
    assert projection.metrics == _literal_metrics(fixture)
    assert projection.journal == _literal_journal_transactions(fixture)
    assert projection.semantic_guards == SEMANTIC_GUARDS
    event_ids: set[str] = set()
    for transaction in projection.journal:
        assert transaction["event_id"] not in event_ids
        event_ids.add(transaction["event_id"])
        assert sum(
            (amount for _, amount in transaction["postings"]), Fraction()
        ) == 0
    expected_roots = _literal_result_roots(fixture)
    assert projection.result["engine_id"] == EXPECTED_ENGINE_ID
    assert projection.result["ledger_bytes"] == _literal_ledger_bytes(fixture)
    assert projection.result["ledger_row_count"] == 3
    assert projection.result["oracle_metrics"] == _literal_metrics(fixture)
    assert projection.result["all_transactions_balanced"] is True
    assert projection.result["journal_transaction_count"] == len(
        projection.journal
    )
    for key, expected in expected_roots.items():
        assert projection.result[key] == expected


def _find_row(projection: AuditProjection, arm: str) -> dict[str, Any]:
    return next(row for row in projection.ledger_rows if row["arm"] == arm)


def _set_row(arm: str, field: str, value: Any) -> Callable[[AuditProjection], None]:
    def mutate(projection: AuditProjection) -> None:
        _find_row(projection, arm)[field] = value

    return mutate


def _set_metric(
    arm: str,
    field: str,
    value: Any,
) -> Callable[[AuditProjection], None]:
    def mutate(projection: AuditProjection) -> None:
        projection.metrics["arms"][arm][field] = value

    return mutate


def _set_guard(field: str, value: Any) -> Callable[[AuditProjection], None]:
    def mutate(projection: AuditProjection) -> None:
        projection.semantic_guards[field] = value

    return mutate


def _drop_journal_leg(projection: AuditProjection) -> None:
    projection.journal[0]["postings"] = projection.journal[0]["postings"][:-1]


def _duplicate_journal_event(projection: AuditProjection) -> None:
    projection.journal[1]["event_id"] = projection.journal[0]["event_id"]


def _post_cumulative_mark_instead_of_delta(projection: AuditProjection) -> None:
    mark = next(
        item
        for item in projection.journal
        if item["arm"] == "EXECUTABLE_BASE" and item["event_kind"] == "MARK_DELTA"
    )
    mark["postings"] = (
        ("UNREALIZED_ASSET", Fraction(-39_996)),
        ("UNREALIZED_PNL", Fraction(39_996)),
    )


def _invert_cluster_connectivity(projection: AuditProjection) -> None:
    projection.metrics["arms"]["EXECUTABLE_BASE"][
        "currency_time_cluster_observations"
    ][0]["currency_nodes"] = ["USD"]


@dataclass(frozen=True)
class MutationCase:
    mutation_id: str
    name: str
    apply: Callable[[AuditProjection], None]


MUTATION_CASES = (
    MutationCase("MK01A", "LONG_ENTRY_BID_INSTEAD_OF_ASK", _set_row(
        "EXECUTABLE_BASE", "entry_price_numerator", 9_999_000_000
    )),
    MutationCase("MK01B", "LONG_EXIT_ASK_INSTEAD_OF_BID", _set_row(
        "EXECUTABLE_BASE", "exit_price_numerator", 10_101_000_000
    )),
    MutationCase("MK01C", "SHORT_ENTRY_ASK_INSTEAD_OF_BID", _set_guard(
        "short_entry_side", "ASK"
    )),
    MutationCase("MK01D", "SHORT_EXIT_BID_INSTEAD_OF_ASK", _set_guard(
        "short_exit_side", "BID"
    )),
    MutationCase("MK02A", "SLIPPAGE_SIGN_REVERSED", _set_row(
        "ADVERSE_STRESS", "latency_spread_slippage_drag_jpy_micros", -39_992
    )),
    MutationCase("MK02B", "PIP_NORMALIZATION_OMITTED", _set_row(
        "ADVERSE_STRESS", "entry_price_numerator", 10_001_000_000
    )),
    MutationCase("MK03A", "ASSET_LIABILITY_EXECUTABLE_SIDE_SWAPPED", _set_row(
        "EXECUTABLE_BASE",
        "signed_currency_exposure_after_entry_jpy_micros",
        {"JPY": 99_980_001, "USD": -99_980_001},
    )),
    MutationCase("MK03B", "INVERSE_CONVERSION_MULTIPLIED_INSTEAD_OF_DIVIDED", _set_guard(
        "inverse_conversion_operation", "MULTIPLY"
    )),
    MutationCase("MK03C", "FUTURE_CONVERSION_QUOTE_ACCEPTED", _set_guard(
        "future_conversion_quote_rejected", False
    )),
    MutationCase("MK03D", "STALE_CONVERSION_QUOTE_ACCEPTED", _set_guard(
        "stale_conversion_quote_rejected", False
    )),
    MutationCase("MK04A", "UNITS_CEILED_INSTEAD_OF_FLOORED", _set_row(
        "EXECUTABLE_BASE", "units_micros", 999_901
    )),
    MutationCase("MK04B", "SIGNED_VALUE_TRUNCATED_TOWARD_ZERO", _set_row(
        "EXECUTABLE_BASE", "economic_net_pnl_jpy_micros_numerator", 195_478_440_202
    )),
    MutationCase("MK04C", "RISK_VALUE_FLOORED_BEFORE_REQUIRED_OUTWARD_ROUND", _set_row(
        "EXECUTABLE_BASE", "marked_or_exit_notional_jpy_micros", 100_979_900
    )),
    MutationCase("MK05A", "COMMISSION_ROUNDED_AFTER_COMBINING_SIDES", _set_row(
        "EXECUTABLE_BASE", "commission_jpy_micros", 2_009
    )),
    MutationCase("MK05B", "ONE_COMMISSION_SIDE_OMITTED", _set_row(
        "EXECUTABLE_BASE", "commission_jpy_micros", 1_000
    )),
    MutationCase("MK05C", "DIRECT_COST_USES_TARGET_NOTIONAL", _set_row(
        "EXECUTABLE_BASE", "commission_jpy_micros", 2_000
    )),
    MutationCase("MK06A", "FINANCING_USES_SOURCE_TIME", _set_guard(
        "financing_clock", "SOURCE"
    )),
    MutationCase("MK06B", "FINANCING_USES_EXIT_NOTIONAL", _set_row(
        "EXECUTABLE_BASE", "financing_jpy_micros", 505
    )),
    MutationCase("MK06C", "FINANCING_USES_WRONG_TERMINAL_CLOCK", _set_guard(
        "terminal_financing_clock", "LAST_QUOTE"
    )),
    MutationCase("MK07A", "ENTRY_ADMITTED_BEFORE_DUE_EXIT", _set_guard(
        "entry_processed_after_due_exits", False
    )),
    MutationCase("MK07B", "OPPOSITE_SAME_PAIR_NETTED_INSTEAD_OF_REJECTED", _set_guard(
        "opposite_same_pair_policy", "NET"
    )),
    MutationCase("MK08A", "CAP_USES_TARGET_NOTIONAL", _set_row(
        "EXECUTABLE_BASE", "gross_open_notional_after_entry_jpy_micros", 100_000_000
    )),
    MutationCase("MK08B", "CURRENCY_NODE_INCIDENCE_SIGN_INVERTED", _set_row(
        "EXECUTABLE_BASE",
        "signed_currency_exposure_after_entry_jpy_micros",
        {"JPY": 99_980_001, "USD": -99_980_001},
    )),
    MutationCase("MK08C", "MARGIN_REQUIREMENT_FLOORED", _set_row(
        "EXECUTABLE_BASE", "required_margin_after_entry_jpy_micros", 4_999_000
    )),
    MutationCase("MK09A", "TERMINAL_LIQUIDATION_USES_MIDPOINT", _set_row(
        "EXECUTABLE_BASE", "exit_price_numerator", 10_100_000_000
    )),
    MutationCase("MK09B", "TERMINAL_LIQUIDATION_OMITS_ACCRUED_COSTS", _set_row(
        "EXECUTABLE_BASE", "financing_jpy_micros", 0
    )),
    MutationCase("MK09C", "REALIZED_AND_UNREALIZED_PNL_DOUBLE_COUNTED", _set_row(
        "EXECUTABLE_BASE", "net_pnl_jpy_micros", 1_957_294
    )),
    MutationCase("MK10A", "ADMISSIONS_CONTINUE_AFTER_CLOSEOUT_OR_RUIN", _set_guard(
        "admissions_halt_after_ruin", False
    )),
    MutationCase("MK10B", "TERMINAL_LIQUIDATION_OMITTED", _set_guard(
        "terminal_liquidation_required", False
    )),
    MutationCase("MK11A", "JOURNAL_POSTING_LEG_DROPPED", _drop_journal_leg),
    MutationCase("MK11B", "DUPLICATE_EVENT_ACCEPTED", _duplicate_journal_event),
    MutationCase(
        "MK11C",
        "CUMULATIVE_BALANCE_POSTED_INSTEAD_OF_DELTA",
        _post_cumulative_mark_instead_of_delta,
    ),
    MutationCase("MK12A", "ECONOMIC_LOT_ROUNDED_PER_TICKET", _set_guard(
        "economic_lot_rounding", "PER_TICKET"
    )),
    MutationCase(
        "MK12B",
        "CURRENCY_CLUSTER_GRAPH_BUILT_WITH_WRONG_CONNECTIVITY",
        _invert_cluster_connectivity,
    ),
    MutationCase("MK12C", "CVAR_TAIL_COUNT_FLOORED", _set_guard(
        "cvar_tail_count_rounding", "FLOOR"
    )),
    MutationCase("MK12D", "DRAWDOWN_USES_WRONG_OBSERVATION_SET", _set_metric(
        "EXECUTABLE_BASE", "max_drawdown_jpy_micros", 0
    )),
)


def test_mutation_inventory_freezes_all_36_named_operators() -> None:
    assert len(MUTATION_CASES) == 36
    assert len({case.mutation_id for case in MUTATION_CASES}) == 36
    assert len({case.name for case in MUTATION_CASES}) == 36
    assert Counter(case.mutation_id[:4] for case in MUTATION_CASES) == {
        "MK01": 4,
        "MK02": 2,
        "MK03": 4,
        "MK04": 3,
        "MK05": 3,
        "MK06": 3,
        "MK07": 2,
        "MK08": 3,
        "MK09": 3,
        "MK10": 2,
        "MK11": 3,
        "MK12": 4,
    }


@pytest.mark.parametrize(
    "case",
    MUTATION_CASES,
    ids=lambda case: f"{case.mutation_id}-{case.name}",
)
def test_named_mutation_operator_has_zero_survivors(
    canonical_fixture: CanonicalFixture,
    case: MutationCase,
) -> None:
    result = reference.replay_reference(dict(canonical_fixture.artifacts))
    projection = _audit_projection(canonical_fixture, result)
    _assert_literal_audit(canonical_fixture, projection)
    case.apply(projection)
    with pytest.raises(AssertionError):
        _assert_literal_audit(canonical_fixture, projection)
