#!/usr/bin/env python3
"""Hand-derived golden ledger for the independent JPY oracle V2.

The economic values in ``ECONOMIC_LITERALS`` were calculated independently
from the oracle and verifier implementations.  This module only performs
canonical serialization, artifact sealing, and ledger hash chaining around
those reviewed literals.  Tests for both implementations must match the exact
ledger bytes and metrics produced here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


START_NS = 1_767_225_600_000_000_000
DAY_NS = 86_400_000_000_000
END_NS = 1_769_904_000_000_000_000
ZERO_SHA = "0" * 64
RATIO_SCALE = 10**18
CLASSIFICATION = "FUTURE_ONLY_ACCOUNTING_ONLY_LOCAL_UNANCHORED_NOT_ADMISSIBLE"
ANCHOR_STATUS = "LOCAL_UNANCHORED"
EXECUTION_PROVENANCE_SCOPE = (
    "LOCAL_CALLER_ASSERTED_CONTENT_BINDING_NOT_EXECUTION_ATTESTATION_"
    "NOT_EXTERNALLY_ANCHORED"
)
RELEASE_CONTENT_BINDING_FIELDS = (
    "code_sha256",
    "contract_sha256",
    "schema_sha256",
    "launcher_sha256",
    "snapshot_mode",
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = digest(canonical(result))
    return result


def ratio_text(numerator: int, denominator: int) -> str:
    """Floor a signed accounting ratio to exactly eighteen decimals."""
    if denominator <= 0:
        raise ValueError("ratio denominator must be positive")
    scaled = numerator * RATIO_SCALE // denominator
    sign = "-" if scaled < 0 else ""
    magnitude = abs(scaled)
    return (
        f"{sign}{magnitude // RATIO_SCALE}."
        f"{magnitude % RATIO_SCALE:018d}"
    )


def nonnegative_ratio_ceiling_text(numerator: int, denominator: int) -> str:
    """Round a nonnegative risk ratio outward to eighteen decimals."""
    if numerator < 0 or denominator <= 0:
        raise ValueError("nonnegative ratio inputs invalid")
    scaled = (numerator * RATIO_SCALE + denominator - 1) // denominator
    return f"{scaled // RATIO_SCALE}.{scaled % RATIO_SCALE:018d}"


# Reviewed economic literals.  Do not replace these values with calls into the
# oracle/verifier or with generated output.  The BASE unit count is
# floor(100,000,000 * 1,000,000 / 100,010,000) = 999,900; ADVERSE uses
# 100,020,000 and yields 999,800.  Entry admission marks those units at the
# executable liquidation side: BASE is 999,900 * 99.99 = 99,980,001 JPY micros
# and ADVERSE is 999,800 * 99.98 = 99,960,004 JPY micros.  Currency exposure
# is a separate signed two-node projection.  The USD base asset converts at the
# causal market BID (99.99), while the JPY quote countervalue uses the arm's
# current liquidation price (RAW 100.00, BASE 99.99, ADVERSE 99.98).  Thus the
# reviewed base/quote node pairs are respectively (+99,990,000,-100,000,000),
# (+99,980,001,-99,980,001), and (+99,970,002,-99,960,004).  Per-side
# commission and financing fractions are rounded upward independently, while
# signed P/L and performance ratios are rounded toward lower equity.
ECONOMIC_LITERALS = {
    "RAW_SIGNAL": {
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
    },
    "EXECUTABLE_BASE": {
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
    },
    "ADVERSE_STRESS": {
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
    },
}


def build_golden_payload() -> dict[str, Any]:
    source = [
        {
            "schema_version": 1,
            "provider_id": "GOLDEN",
            "instrument": "USD_JPY",
            "bid_ticks": 9_999,
            "ask_ticks": 10_001,
            "tick_scale": 100,
            "source_ts_ns": START_NS,
            "arrival_ts_ns": START_NS,
            "provider_event_id": "g1",
            "sequence": 1,
            "heartbeat": False,
            "quality_flags": [],
        },
        {
            "schema_version": 1,
            "provider_id": "GOLDEN",
            "instrument": "USD_JPY",
            "bid_ticks": 9_999,
            "ask_ticks": 10_001,
            "tick_scale": 100,
            "source_ts_ns": START_NS + 1_000_000_000,
            "arrival_ts_ns": START_NS + 1_000_000_000,
            "provider_event_id": "g2",
            "sequence": 2,
            "heartbeat": False,
            "quality_flags": [],
        },
        {
            "schema_version": 1,
            "provider_id": "GOLDEN",
            "instrument": "USD_JPY",
            "bid_ticks": 10_099,
            "ask_ticks": 10_101,
            "tick_scale": 100,
            "source_ts_ns": START_NS + DAY_NS + 1_000_000_000,
            "arrival_ts_ns": START_NS + DAY_NS + 1_000_000_000,
            "provider_event_id": "g3",
            "sequence": 3,
            "heartbeat": False,
            "quality_flags": [],
        },
    ]
    source_blob = b"".join(canonical(row) + b"\n" for row in source)
    prefix = ZERO_SHA
    enriched: list[dict[str, Any]] = []
    for row in source:
        event_hash = digest(canonical(row) + b"\n")
        prefix = digest(canonical({
            "previous_hash": prefix,
            "source_event_sha256": event_hash,
        }))
        enriched.append({
            **row,
            "source_event_sha256": event_hash,
            "source_prefix_root_sha256": prefix,
        })

    registry = seal({
        "schema_version": 1,
        "registry_id": "FROZEN_FX_INSTRUMENT_REGISTRY_V1",
        "instruments": {"USD_JPY": {"pip_ticks": 1, "price_scale": 100}},
    }, "registry_sha256")
    source_manifest = seal({
        "schema_version": 2,
        "source_bytes_sha256": digest(source_blob),
        "source_size_bytes": len(source_blob),
        "event_count": 3,
        "first_source_ts_ns": START_NS,
        "last_source_ts_ns": START_NS + DAY_NS + 1_000_000_000,
        "provider_allowlist": ["GOLDEN"],
        "instrument_registry_sha256": registry["registry_sha256"],
        "stream_policies": [{
            "provider_id": "GOLDEN",
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
    proposal_row = {
        "proposal_ordinal": 1,
        "decision_source_ts_ns": START_NS,
        "decision_arrival_ts_ns": START_NS,
        "available_at_ns": START_NS,
        "decision_source_event_sha256": enriched[0]["source_event_sha256"],
        "completed_data_watermark_source_ts_ns": START_NS,
        "completed_data_prefix_root_sha256": enriched[0]["source_prefix_root_sha256"],
        "instrument": "USD_JPY",
        "direction": 1,
        "notional_jpy_micros": 100_000_000,
        "max_age_ns": DAY_NS,
        "worker_key": "GOLDEN_FIXED",
        "action": "ENTER",
    }
    proposal = seal({
        "schema_version": 2,
        "candidate_key": "GOLDEN-USDJPY-LONG",
        "provenance": provenance,
        "rows": [proposal_row],
    }, "proposal_sha256")
    execution = seal({
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
    inventory = seal({
        "schema_version": 2,
        "policy_id": "FROZEN_INVENTORY_POLICY_V2",
        "max_gross_notional_jpy_micros": 1_000_000_000,
        "max_currency_notional_jpy_micros": 1_000_000_000,
        "max_open_positions": 1,
        "same_pair_collision": "REJECT_NEW",
        "terminal_liquidation": True,
    }, "inventory_policy_sha256")
    accounting = seal({
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
    evaluation = seal({
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
    authority = seal({
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

    signal_id = digest(canonical({
        "candidate_key": "GOLDEN-USDJPY-LONG",
        "proposal_ordinal": 1,
        "decision_source_ts_ns": START_NS,
        "decision_arrival_ts_ns": START_NS,
        "decision_source_event_sha256": enriched[0]["source_event_sha256"],
        "completed_data_prefix_root_sha256": enriched[0]["source_prefix_root_sha256"],
        "instrument": "USD_JPY",
        "direction": 1,
        "notional_jpy_micros": 100_000_000,
        "max_age_ns": DAY_NS,
        "worker_key": "GOLDEN_FIXED",
        "detector_code_sha256": "1" * 64,
        "detector_policy_sha256": "2" * 64,
        "generator_policy_sha256": "3" * 64,
    }))
    economic_lot_id = digest(canonical({
        "candidate_key": "GOLDEN-USDJPY-LONG",
        "decision_source_ts_ns": START_NS,
        "decision_arrival_ts_ns": START_NS,
        "decision_source_event_sha256": enriched[0]["source_event_sha256"],
        "completed_data_prefix_root_sha256": enriched[0]["source_prefix_root_sha256"],
        "instrument": "USD_JPY",
        "direction": 1,
        "target_notional_jpy_micros": 100_000_000,
        "max_age_ns": DAY_NS,
        "worker_key": "GOLDEN_FIXED",
        "detector_code_sha256": "1" * 64,
        "detector_policy_sha256": "2" * 64,
        "generator_policy_sha256": "3" * 64,
    }))
    entry, exit_event = enriched[1], enriched[2]

    def source_reference(event: dict[str, Any]) -> dict[str, Any]:
        return {
            "provider_id": "GOLDEN",
            "source_event_sha256": event["source_event_sha256"],
            "source_ts_ns": event["source_ts_ns"],
            "arrival_ts_ns": event["arrival_ts_ns"],
            "execution_policy_sha256": execution["execution_policy_sha256"],
        }

    dispositions: list[dict[str, Any]] = []
    for arm in ("RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS"):
        value = ECONOMIC_LITERALS[arm]
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
            "latency_spread_slippage_drag_jpy_micros": value[
                "execution_drag"
            ],
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
            "gross_open_notional_after_entry_jpy_micros": value[
                "entry_exposure"
            ],
            "marked_equity_after_entry_jpy_micros": value["entry_equity"],
            "required_margin_after_entry_jpy_micros": value[
                "required_margin"
            ],
            "free_margin_after_entry_jpy_micros": value["free_margin"],
            "entry_source_reference": source_reference(entry),
            "exit_source_reference": source_reference(exit_event),
            "terminal_inventory_mtm_jpy_micros": 0,
            "external_order_count": 0,
        })

    previous = ZERO_SHA
    ledger: list[dict[str, Any]] = []
    for sequence, disposition in enumerate(dispositions, 1):
        row = {
            "ledger_schema_version": 2,
            "ledger_sequence": sequence,
            "previous_hash": previous,
            **disposition,
        }
        row["record_hash"] = digest(canonical(row))
        previous = row["record_hash"]
        ledger.append(row)
    ledger_bytes = b"".join(canonical(row) + b"\n" for row in ledger)

    signal_set_hash = digest(canonical([signal_id]))
    economic_lot_set_hash = digest(canonical([economic_lot_id]))
    time_bucket = 1_767_225_601
    cluster_id = digest(canonical({
        "time_bucket": time_bucket,
        "currency_nodes": ["JPY", "USD"],
    }))
    arm_metrics: dict[str, Any] = {}
    for arm in ("RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS"):
        value = ECONOMIC_LITERALS[arm]
        arm_metrics[arm] = {
            "proposal_count": 1,
            "executed_count": 1,
            "disposition_counts": {"FILLED_CLOSED": 1},
            "signal_id_set_sha256": signal_set_hash,
            "common_gross_pnl_jpy_micros": 1_000_000,
            "realized_cost_jpy_micros": value["cost"],
            "fill_sizing_drag_jpy_micros": value["fill_sizing_drag"],
            "latency_spread_slippage_drag_jpy_micros": value[
                "execution_drag"
            ],
            "direct_commission_financing_cost_jpy_micros": (
                value["commission"] + value["financing"]
            ),
            "admission_opportunity_drag_jpy_micros": 0,
            "total_execution_and_admission_drag_jpy_micros": value["cost"],
            "net_pnl_jpy_micros": value["net"],
            "ending_equity_jpy_micros": value["ending_equity"],
            "ending_equity_multiple": ratio_text(
                value["ending_equity"], 1_000_000_000
            ),
            "direction_accuracy": "1.000000000000000000",
            "max_drawdown_jpy_micros": value["drawdown"],
            "max_drawdown_ratio": nonnegative_ratio_ceiling_text(
                value["drawdown"], 1_000_000_000
            ),
            "cvar_tail_bps": 500,
            "cluster_cvar_jpy_micros": value["net"],
            "cluster_cvar_return": ratio_text(
                value["economic_net_numerator"],
                value["economic_net_denominator"] * 1_000_000_000,
            ),
            "currency_time_cluster_n_eff": 1,
            "currency_time_cluster_observations": [{
                "cluster_id": cluster_id,
                "time_bucket": time_bucket,
                "currency_nodes": ["JPY", "USD"],
                "source_signal_set_sha256": economic_lot_set_hash,
                "ledger_net_pnl_jpy_micros": value["net"],
                "cluster_risk_net_pnl_jpy_micros": value["net"],
                "signed_return": ratio_text(
                    value["net"], 1_000_000_000
                ),
            }],
            "monthly": [{
                "month_id": "2026-01",
                "comparable_full_month": True,
                "segment_start_ts_ns": START_NS,
                "segment_end_ts_ns": END_NS,
                "start_equity_jpy_micros": 1_000_000_000,
                "end_equity_jpy_micros": value["ending_equity"],
                "equity_multiple": ratio_text(
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
    metrics["metrics_sha256"] = digest(canonical(metrics))
    for value in ECONOMIC_LITERALS.values():
        assert value["target_notional"] >= value["filled_notional"]
        assert value["filled_notional"] == value["financing_basis_notional"]
        assert value["cost"] == 1_000_000 - value["net"]
        assert value["cost"] == (
            value["fill_sizing_drag"]
            + value["execution_drag"]
            + value["commission"]
            + value["financing"]
        )
        assert value["ending_equity"] == 1_000_000_000 + value["net"]
        assert (
            value["economic_net_numerator"]
            // value["economic_net_denominator"]
        ) == value["net"]
        assert value["base_node_exposure"] > 0
        assert value["quote_node_exposure"] < 0
    return {
        "inputs": {
            "source_blob_utf8": source_blob.decode("utf-8"),
            "instrument_registry": registry,
            "source_manifest": source_manifest,
            "proposal": proposal,
            "execution_policy": execution,
            "inventory_policy": inventory,
            "accounting_policy": accounting,
            "evaluation_policy": evaluation,
            "authority_policy": authority,
        },
        "expected": {
            "ledger_utf8": ledger_bytes.decode("utf-8"),
            "ledger_sha256": digest(ledger_bytes),
            "ledger_size_bytes": len(ledger_bytes),
            "oracle_metrics": metrics,
            "manifest_contract": {
                "oracle_implementation": "INDEPENDENT_JPY_ORACLE_V2",
                "status": "COMPLETE",
                "classification": CLASSIFICATION,
                "anchor_status": ANCHOR_STATUS,
                "causal_signal_admission": False,
                "release_evidence_eligible": False,
                "detector_replay_receipt_required": True,
                "producer_result_or_metrics_used": False,
                "proposal_identity_generated_by_oracle": True,
                "terminal_inventory_mtm_jpy_micros": 0,
                "external_orders": 0,
                "oracle_release_content_binding": {
                    "required_fields": list(RELEASE_CONTENT_BINDING_FIELDS),
                    "values_are_exact_trusted_snapshot_bindings": True,
                },
                "oracle_execution_provenance_scope": (
                    EXECUTION_PROVENANCE_SCOPE
                ),
            },
            "verifier_receipt_contract": {
                "classification": CLASSIFICATION,
                "anchor_status": ANCHOR_STATUS,
                "causal_signal_admission": False,
                "release_evidence_eligible": False,
                "admission_eligible": False,
                "oracle_release_content_binding_preserved_exactly": True,
                "oracle_execution_provenance_scope_preserved_exactly": True,
            },
        },
    }


def main() -> int:
    print(canonical(build_golden_payload()).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
