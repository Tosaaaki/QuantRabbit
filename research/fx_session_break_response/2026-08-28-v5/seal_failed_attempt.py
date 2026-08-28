#!/usr/bin/env python3
"""Seal the single V5 fail-closed replay attempt without rerunning it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNNER_SHA256 = "a8a83d6ec8862907b8b36211bd8f94331e9857d967d13d423236c89545188632"
PREREG_SHA256 = "6542095c980ec421d779f21b909dec8f860c5b745a349c8dbafea904d1be4b7e"
PREREG_CANONICAL_SHA256 = "37981d1cf749d397f4626aa60c379ddf63e69a700d4278647e078bd2872802ac"
DATASET_SHA256 = "721904751fc1d590a64c7cefd0a533e7df314f043b10783c116d2a82793f14fb"
ZERO_OBSERVATION_CONFIGS = [
    "LONDON_MIDDAY__REJECT_FADE__D50__G50__BANY__AMODE_MATCHED__H24",
    "LONDON_MIDDAY__REJECT_FADE__D50__G50__BANY__AMODE_MATCHED__H48",
    "LONDON_MIDDAY__REJECT_FADE__D50__G50__BMODE_MATCHED__AMODE_MATCHED__H24",
    "LONDON_MIDDAY__REJECT_FADE__D50__G50__BMODE_MATCHED__AMODE_MATCHED__H48",
    "LONDON_MIDDAY__REJECT_FADE__D50__G67__BANY__AMODE_MATCHED__H24",
    "LONDON_MIDDAY__REJECT_FADE__D50__G67__BANY__AMODE_MATCHED__H48",
    "LONDON_MIDDAY__REJECT_FADE__D50__G67__BMODE_MATCHED__AANY__H24",
    "LONDON_MIDDAY__REJECT_FADE__D50__G67__BMODE_MATCHED__AANY__H48",
    "LONDON_MIDDAY__REJECT_FADE__D50__G67__BMODE_MATCHED__AMODE_MATCHED__H24",
    "LONDON_MIDDAY__REJECT_FADE__D50__G67__BMODE_MATCHED__AMODE_MATCHED__H48",
    "LONDON_MIDDAY__REJECT_FADE__D67__G50__BANY__AMODE_MATCHED__H24",
    "LONDON_MIDDAY__REJECT_FADE__D67__G50__BANY__AMODE_MATCHED__H48",
    "LONDON_MIDDAY__REJECT_FADE__D67__G50__BMODE_MATCHED__AMODE_MATCHED__H24",
    "LONDON_MIDDAY__REJECT_FADE__D67__G50__BMODE_MATCHED__AMODE_MATCHED__H48",
    "LONDON_MIDDAY__REJECT_FADE__D67__G67__BANY__AMODE_MATCHED__H24",
    "LONDON_MIDDAY__REJECT_FADE__D67__G67__BANY__AMODE_MATCHED__H48",
    "LONDON_MIDDAY__REJECT_FADE__D67__G67__BMODE_MATCHED__AANY__H24",
    "LONDON_MIDDAY__REJECT_FADE__D67__G67__BMODE_MATCHED__AANY__H48",
    "LONDON_MIDDAY__REJECT_FADE__D67__G67__BMODE_MATCHED__AMODE_MATCHED__H24",
    "LONDON_MIDDAY__REJECT_FADE__D67__G67__BMODE_MATCHED__AMODE_MATCHED__H48",
]


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    if len(ZERO_OBSERVATION_CONFIGS) != 20 or len(set(ZERO_OBSERVATION_CONFIGS)) != 20:
        raise RuntimeError("failed-config receipt must contain exactly 20 unique rows")
    if sha_file(ROOT / "replay_session_break_response.py") != RUNNER_SHA256:
        raise RuntimeError("executed runner bytes changed before failure seal")
    if sha_file(ROOT / "PREREGISTRATION.json") != PREREG_SHA256:
        raise RuntimeError("preregistration bytes changed before failure seal")
    result_path = ROOT / "result.json"
    packet_path = ROOT / "evidence_packet.json"
    if result_path.exists() or packet_path.exists():
        raise RuntimeError("failure evidence already sealed")
    unavailable = {
        "mean_pips": None,
        "total_pips": None,
        "total_jpy": None,
        "equity_multiple": None,
        "reason": "family standardization failed before winner selection",
    }
    result = {
        "schema": "QR_FX_SESSION_BREAK_RESPONSE_RESULT_V1",
        "candidate_id": "FX_SESSION_BREAK_RESPONSE_SURFACE_V5",
        "status": "REJECTED_DISCOVERY_FAMILY_UNSTANDARDIZABLE",
        "reason_code": "MAX_T_ALL_128_STANDARDIZATION_FAILED",
        "profit_proven": False,
        "strategy_admitted": False,
        "future_holdout_required": True,
        "attempt": {
            "attempt_number": 1,
            "started_command": "python3 replay_session_break_response.py --run-once",
            "finished_at_utc": "2026-08-28T06:53:13.000000Z",
            "exit_code": 1,
            "elapsed_real_seconds": 7.39,
            "rerun_permitted_for_this_candidate": False,
        },
        "authority": {
            "network_attempts": 0,
            "credential_reads": 0,
            "external_order_attempts": 0,
            "external_orders": 0,
            "broker_mutations": 0,
            "launchd_actions": 0,
            "git_actions": 0,
        },
        "hashes": {
            "preregistration_sha256": PREREG_SHA256,
            "preregistration_canonical_sha256": PREREG_CANONICAL_SHA256,
            "executed_runner_sha256": RUNNER_SHA256,
            "dataset_sha256": DATASET_SHA256,
        },
        "decode_audit": {
            "discovery_prefix_rows_decoded": 166702,
            "winner_locked": False,
            "validation_rows_decoded": 0,
            "post_boundary_price_or_volume_rows_decoded": 0,
            "post_boundary_label_rows_computed": 0,
            "holdout_rows_decoded": 0,
        },
        "discovery_family_standardization": {
            "family_count_required": 128,
            "configs_with_at_least_one_observation": 108,
            "zero_observation_config_count": 20,
            "zero_observation_config_ids": ZERO_OBSERVATION_CONFIGS,
            "standardized_count": None,
            "standardized_count_required": 128,
            "max_t_fwer_computed": False,
            "corrected_lcb_pips": None,
            "winner_selected": False,
        },
        "locked_internal_validation": {
            "decoded": False,
            "trades": None,
            "N_eff": None,
            "pair_results": None,
            "anchored_month_results": None,
            "terminal_inventory": None,
            "reason": "winner selection failed before validation byte decode",
        },
        "execution_arms": {
            "RAW_SIGNAL": dict(unavailable),
            "EXECUTABLE_BASE": dict(unavailable),
            "ADVERSE_STRESS": dict(unavailable),
        },
        "interpretation": {
            "gross_edge_evaluated": False,
            "cost_edge_evaluated": False,
            "interaction_edge_evaluated": False,
            "profit_claim_allowed": False,
            "next_version_rule": "A materially new preregistered family may address zero-density cells; V5 thresholds and family remain immutable and rejected.",
        },
    }
    result["result_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    packet = {
        "schema": "QR_FX_SESSION_BREAK_RESPONSE_EVIDENCE_PACKET_V1",
        "candidate_id": result["candidate_id"],
        "status": result["status"],
        "reason_code": result["reason_code"],
        "attempt_number": 1,
        "attempt_exit_code": 1,
        "result_sha256": result["result_sha256"],
        "preregistration_sha256": PREREG_SHA256,
        "executed_runner_sha256": RUNNER_SHA256,
        "dataset_sha256": DATASET_SHA256,
        "family_count_required": 128,
        "zero_observation_config_count": 20,
        "validation_rows_decoded": 0,
        "holdout_rows_decoded": 0,
        "profit_proven": False,
        "strategy_admitted": False,
        "network_attempts": 0,
        "credential_reads": 0,
        "external_orders": 0,
    }
    packet["packet_sha256"] = hashlib.sha256(canonical(packet)).hexdigest()
    atomic_json(result_path, result)
    atomic_json(packet_path, packet)
    print(json.dumps({
        "status": result["status"],
        "attempt": 1,
        "exit_code": 1,
        "zero_observation_configs": 20,
        "validation_rows_decoded": 0,
        "result_sha256": result["result_sha256"],
        "packet_sha256": packet["packet_sha256"],
        "external_orders": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
