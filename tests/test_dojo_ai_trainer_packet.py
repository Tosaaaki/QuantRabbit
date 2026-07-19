from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from quant_rabbit.dojo_ai_trainer_packet import (
    DojoAITrainerPacketError,
    build_trainer_packet,
    canonical_packet_sha256,
    verify_trainer_packet,
)
from quant_rabbit.dojo_ai_tuning_state import initialize_tuning_state
from quant_rabbit.dojo_bot_catalog import bot_config_risk_vector
from quant_rabbit.dojo_bot_trainer import (
    CELL_CONTRACT,
    EVALUATION_CONTRACT,
    PROPOSAL_CONTRACT,
    STUDY_CONTRACT,
    _score_candidate,
    seal_candidate_proposal,
    seal_study,
)
from quant_rabbit.dojo_candidate_lineage_registry import (
    bind_result,
    initialize_registry,
    seal_study_attempt,
)


PAIRS = ["CAD_JPY", "EUR_USD", "GBP_USD", "USD_JPY"]
SOURCES = {
    "bots/lab_bot.py": "a" * 64,
    "src/quant_rabbit/dojo_bot_trainer.py": "b" * 64,
}


def _canonical_sha(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(root: Path, name: str, value: object) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _sealed_study() -> dict:
    proposal = seal_candidate_proposal(
        {
            "contract": PROPOSAL_CONTRACT,
            "schema_version": 1,
            "candidate_id": "qr-a1-c01",
            "family": "spike_fade",
            "hypothesis": "Cost-sensitive shock reversion with fixed risk.",
            "config": {
                "signal": "spike_fade",
                "pairs": PAIRS,
                "tp_atr": 3.0,
                "sl_pips": 25.0,
                "ceiling_min": 60,
                "max_concurrent_per_pair": 1,
                "global_max_concurrent": 4,
                "per_pos_lev": 5.0,
                "atr_floor_pips": 0.5,
            },
            "risk_increase": False,
        }
    )
    study = {
        "contract": STUDY_CONTRACT,
        "schema_version": 1,
        "study_id": "qr-trainer-packet-study-1",
        "window_role": "TRAIN",
        "initial_balance_jpy": 200_000.0,
        "trade_pairs": PAIRS,
        "feed_pairs": PAIRS,
        "candidates": [proposal],
        "window": {
            "start_utc": "2025-03-01T00:00:00Z",
            "end_utc": "2025-04-01T00:00:00Z",
            "corpus_id": "worn-fx-march-2025",
            "corpus_sha256": "1" * 64,
            "evidence_tier": "WORN_TRAIN",
        },
        "cost_arms": {
            "BASE": {
                "slippage_pips_per_fill": 0.0,
                "financing_pips_per_day": 0.0,
                "recorded_spread_multiplier": 1.0,
            },
            "STRESS": {
                "slippage_pips_per_fill": 0.3,
                "financing_pips_per_day": 0.8,
                "recorded_spread_multiplier": 1.0,
            },
        },
        "proposer_evidence": {
            "prompt_sha256": "2" * 64,
            "input_sha256": "3" * 64,
            "raw_response_sha256": "4" * 64,
            "model_claim": "gpt-test",
            "provider_attestation": "UNVERIFIED",
        },
        "search_budget": {
            "attempt_ordinal": 1,
            "total_attempts_in_lineage": 3,
            "max_candidates": 1,
        },
        "thresholds": {
            "normal_mtm_drawdown_max": 0.10,
            "stress_mtm_drawdown_max": 0.15,
            "peak_margin_usage_max": 0.45,
            "margin_reject_rate_max": 0.10,
            "cost_retention_min": 0.50,
            "pair_positive_share_max": 0.50,
            "pair_hhi_max": 0.40,
        },
    }
    return seal_study(study, SOURCES)


def _evaluation(sealed: dict) -> dict:
    proposal = sealed["study"]["candidates"][0]
    risk = bot_config_risk_vector(
        proposal["config"],
        stress_slippage_pips_per_fill=sealed["study"]["cost_arms"]["STRESS"][
            "slippage_pips_per_fill"
        ],
    )
    failed = [
        f"{path}:{arm}:MAIN_REPLAY_FAILED"
        for path in ("OHLC", "OLHC")
        for arm in ("BASE", "STRESS")
    ]
    body = {
        "contract": EVALUATION_CONTRACT,
        "schema_version": 1,
        "study_sha256": sealed["study_sha256"],
        "classification": "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY",
        "fixed_denominator": {
            "candidate_count": 1,
            "intrabar_paths": ["OHLC", "OLHC"],
            "cost_arms": ["BASE", "STRESS"],
            "expected_cell_count": 4,
            "observed_cell_count": 4,
            "coordinate_receipts_complete": True,
            "execution_success_complete": False,
        },
        "candidate_evaluations": [
            {
                "candidate_id": proposal["candidate_id"],
                "status": "TRAIN_REJECT",
                "diagnostic_rank_eligible": False,
                "diagnostic_score": None,
                "risk_policy_receipt": risk,
                "proposer_risk_claim_ignored": True,
                "gate_blockers": [
                    *risk["blocker_codes"],
                    "RUNNER_CELL_FAILURE",
                    "CONTINUOUS_MTM_EVIDENCE_INCOMPLETE",
                    "TERMINAL_EXPOSURE",
                    "ZERO_FILLS_IN_FIXED_CELL",
                    "NON_POSITIVE_COORDINATE_WORST_NET",
                    "NORMAL_DRAWDOWN_TOO_HIGH",
                    "STRESS_DRAWDOWN_TOO_HIGH",
                    "MARGIN_REJECT_RATE_TOO_HIGH",
                    "COST_RETENTION_TOO_LOW",
                    "PAIR_POSITIVE_CONTRIBUTION_TOO_CONCENTRATED",
                    "PAIR_CONTRIBUTION_HHI_TOO_HIGH",
                    "LEAVE_ONE_PAIR_OUT_NOT_POSITIVE",
                    "COUNTERFACTUAL_LOPO_INCOMPLETE",
                    "CAPITAL_LOCK_METRIC_INCOMPLETE",
                ],
                "failed_coordinates": failed,
                "coordinate_worst": {
                    "terminal_net_jpy": 0.0,
                    "realized_max_drawdown_fraction": 0.0,
                    "normal_effective_drawdown_fraction": 1.0,
                    "stress_effective_drawdown_fraction": 1.0,
                    "peak_margin_usage_fraction": 0.0,
                    "margin_reject_rate": 1.0,
                    "cost_retention": None,
                    "pair_positive_share": 1.0,
                    "pair_hhi": 1.0,
                    "effective_positive_pairs": 1.0,
                    "leave_one_pair_out_net_jpy": 0.0,
                    "capital_productivity_per_margin_day": None,
                },
                "cost_retention_by_intrabar": {"OHLC": None, "OLHC": None},
                "capital_productivity_by_cell": {
                    "OHLC:BASE": None,
                    "OHLC:STRESS": None,
                    "OLHC:BASE": None,
                    "OLHC:STRESS": None,
                },
                "mtm_complete": False,
                "mtm_incomplete_uses_realized_dd_for_train_diagnostic_only": False,
                "lopo_replay_complete": False,
                "promotion_gate_passed": False,
                "promotion_blockers": [
                    "WORN_HISTORICAL_TRAIN_ONLY",
                    "PROSPECTIVE_FORWARD_EVIDENCE_REQUIRED",
                    "CONTINUOUS_MTM_EVIDENCE_INCOMPLETE",
                    "COUNTERFACTUAL_LOPO_INCOMPLETE",
                ],
                "proof_eligible": False,
                "promotion_eligible": False,
                "live_permission": False,
                "order_authority": "NONE",
            }
        ],
        "diagnostic_ranking": [],
        "rank_eligible_candidate_ids": [],
        "unranked_candidate_ids": [proposal["candidate_id"]],
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    return {**body, "evaluation_sha256": _canonical_sha(body)}


def _cells(sealed: dict) -> list[dict]:
    proposal = sealed["study"]["candidates"][0]
    zeros = {pair: 0.0 for pair in PAIRS}
    ledger = {
        "main": {
            "artifact_relpath": "sessions/never-opened/ledger.jsonl",
            "artifact_size_bytes": 99,
            "artifact_sha256": "9" * 64,
        }
    }
    rows = []
    for path in ("OHLC", "OLHC"):
        for arm in ("BASE", "STRESS"):
            body = {
                "contract": CELL_CONTRACT,
                "schema_version": 1,
                "study_sha256": sealed["study_sha256"],
                "candidate_id": proposal["candidate_id"],
                "proposal_sha256": proposal["proposal_sha256"],
                "intrabar": path,
                "cost_arm": arm,
                "metrics": {
                    "terminal_net_jpy": 0.0,
                    "terminal_flat": False,
                    "margin_closeouts": 0,
                    "realized_max_drawdown_fraction": 0.0,
                    "mtm_complete": False,
                    "mtm_max_drawdown_fraction": None,
                    "peak_margin_usage_fraction": 0.0,
                    "fill_count": 0,
                    "margin_reject_count": 0,
                    "capital_lock_margin_jpy_hours": 0.0,
                    "pair_pnl_jpy": zeros,
                    "leave_one_pair_out_net_jpy": zeros,
                    "lopo_replay_complete": False,
                },
                "ledger_evidence": ledger,
                "execution_status": "FAILED",
                "failure_code": "MAIN_REPLAY_FAILED",
            }
            rows.append({**body, "cell_sha256": _canonical_sha(body)})
    return rows


def _successful_cells(sealed: dict) -> list[dict]:
    proposal = sealed["study"]["candidates"][0]
    terminal_by_coordinate = {
        ("OHLC", "BASE"): 10_000.0,
        ("OHLC", "STRESS"): 8_000.0,
        ("OLHC", "BASE"): 12_000.0,
        ("OLHC", "STRESS"): 9_600.0,
    }
    rows = []
    for path in ("OHLC", "OLHC"):
        for arm in ("BASE", "STRESS"):
            terminal = terminal_by_coordinate[(path, arm)]
            pair_pnl = {pair: terminal / len(PAIRS) for pair in PAIRS}
            body = {
                "contract": CELL_CONTRACT,
                "schema_version": 1,
                "study_sha256": sealed["study_sha256"],
                "candidate_id": proposal["candidate_id"],
                "proposal_sha256": proposal["proposal_sha256"],
                "intrabar": path,
                "cost_arm": arm,
                "metrics": {
                    "terminal_net_jpy": terminal,
                    "terminal_flat": True,
                    "margin_closeouts": 0,
                    "realized_max_drawdown_fraction": 0.02,
                    "mtm_complete": True,
                    "mtm_max_drawdown_fraction": (0.03 if arm == "BASE" else 0.04),
                    "peak_margin_usage_fraction": 0.20,
                    "fill_count": 100,
                    "margin_reject_count": 1,
                    "capital_lock_margin_jpy_hours": terminal * 24.0,
                    "pair_pnl_jpy": pair_pnl,
                    "leave_one_pair_out_net_jpy": {pair: 2_000.0 for pair in PAIRS},
                    "lopo_replay_complete": True,
                },
                "ledger_evidence": {
                    "main": {
                        "artifact_relpath": "sessions/never-opened/ledger.jsonl",
                        "artifact_size_bytes": 99,
                        "artifact_sha256": "9" * 64,
                    }
                },
                "execution_status": "SUCCESS",
                "failure_code": None,
            }
            rows.append({**body, "cell_sha256": _canonical_sha(body)})
    return rows


def _evaluation_from_cells(sealed: dict, cells: list[dict]) -> dict:
    proposal = sealed["study"]["candidates"][0]
    candidate = _score_candidate(proposal["candidate_id"], cells, sealed_study=sealed)
    ranked = [proposal["candidate_id"]] if candidate["diagnostic_rank_eligible"] else []
    body = {
        "contract": EVALUATION_CONTRACT,
        "schema_version": 1,
        "study_sha256": sealed["study_sha256"],
        "classification": "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY",
        "fixed_denominator": {
            "candidate_count": 1,
            "intrabar_paths": ["OHLC", "OLHC"],
            "cost_arms": ["BASE", "STRESS"],
            "expected_cell_count": 4,
            "observed_cell_count": 4,
            "coordinate_receipts_complete": True,
            "execution_success_complete": all(
                cell["execution_status"] == "SUCCESS" for cell in cells
            ),
        },
        "candidate_evaluations": [candidate],
        "diagnostic_ranking": ranked,
        "rank_eligible_candidate_ids": ranked,
        "unranked_candidate_ids": ([] if ranked else [proposal["candidate_id"]]),
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    return {**body, "evaluation_sha256": _canonical_sha(body)}


def _run(sealed: dict, evaluation: dict, cells: list[dict]) -> dict:
    coordinates = []
    for index, cell in enumerate(cells):
        success = cell["execution_status"] == "SUCCESS"
        coordinates.append(
            {
                "candidate_id": cell["candidate_id"],
                "intrabar": cell["intrabar"],
                "cost_arm": cell["cost_arm"],
                "status": "COMPLETE" if success else "MAIN_REPLAY_FAILED_SENTINEL",
                "main_session_dir": f"sessions/{index}",
                "main_error": None if success else "fixture failure",
                "lopo_replay_complete": success,
                "lopo": (
                    [
                        {
                            "held_out_pair": pair,
                            "status": "VALID_COUNTERFACTUAL_REPLAY",
                            "terminal_net_jpy": cell["metrics"][
                                "leave_one_pair_out_net_jpy"
                            ][pair],
                            "session_dir": f"sessions/{index}-lopo-{pair}",
                            "ledger_path": f"sessions/{index}-lopo-{pair}/ledger.jsonl",
                            "corpus_sha256": "1" * 64,
                        }
                        for pair in PAIRS
                    ]
                    if success
                    else []
                ),
                "cell_sha256": cell["cell_sha256"],
            }
        )
    failed_count = sum(cell["execution_status"] == "FAILED" for cell in cells)
    body = {
        "contract": "QR_DOJO_BOT_TRAINER_RUN_V1",
        "schema_version": 1,
        "study_sha256": sealed["study_sha256"],
        "status": "COMPLETE" if failed_count == 0 else "COMPLETE_WITH_FAILED_CELLS",
        "corpus": {"corpus_sha256": "1" * 64},
        "fixed_denominator": {
            "expected_cell_count": 4,
            "observed_cell_count": 4,
            "failed_cell_count": failed_count,
            "dropped_cell_count": 0,
            "coordinate_receipts_complete": True,
            "execution_success_complete": failed_count == 0,
        },
        "coordinates": coordinates,
        "cells_path": "/not/read/cells.json",
        "evaluation_path": "/not/read/evaluation.json",
        "evaluation_sha256": evaluation["evaluation_sha256"],
        "classification": "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY",
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    return {**body, "run_sha256": _canonical_sha(body)}


def _bind_inputs(
    tmp_path: Path, *, sealed: dict, evaluation: dict, cells: list[dict]
) -> dict:
    run = _run(sealed, evaluation, cells)
    events = tmp_path / "events"
    snapshot = initialize_registry(
        events,
        artifact_root=tmp_path,
        registry_id="qr-trainer-packet-registry",
        lineage_prefix="qr-",
        created_by="pytest",
        event_at_utc="2026-07-20T00:00:00Z",
    )
    study_path = _write_json(tmp_path, "artifacts/study-1.json", sealed)
    snapshot = seal_study_attempt(
        events,
        artifact_root=tmp_path,
        sealed_study_path=study_path,
        expected_tip_sha256=snapshot.latest_event_sha256,
        event_at_utc="2026-07-20T00:00:01Z",
    )
    evaluation_path = _write_json(tmp_path, "artifacts/evaluation-1.json", evaluation)
    snapshot = bind_result(
        events,
        artifact_root=tmp_path,
        evaluation_path=evaluation_path,
        expected_tip_sha256=snapshot.latest_event_sha256,
        event_at_utc="2026-07-20T00:00:02Z",
    )
    tuning_state = initialize_tuning_state(
        events, artifact_root=tmp_path, sealed_study=sealed
    )
    return {
        "run": run,
        "evaluation": evaluation,
        "cells": cells,
        "lineage_events_dir": events,
        "artifact_root": tmp_path,
        "tuning_state": tuning_state,
    }


@pytest.fixture
def packet_inputs(tmp_path: Path) -> dict:
    sealed = _sealed_study()
    cells = _cells(sealed)
    return _bind_inputs(
        tmp_path, sealed=sealed, evaluation=_evaluation(sealed), cells=cells
    )


def _build(inputs: dict, **overrides) -> dict:
    arguments = {**inputs, **overrides}
    return build_trainer_packet(**arguments)


def test_packet_contains_full_grid_unknowns_and_no_raw_references(
    packet_inputs: dict,
) -> None:
    drive = [
        {
            "artifact_kind": "REPORT",
            "drive_file_id": "driveFile123",
            "content_sha256": "6" * 64,
            "content_size_bytes": 321,
            "remote_verified": True,
            "metadata_receipt_sha256": "7" * 64,
        }
    ]
    packet = _build(packet_inputs, drive_evidence_refs=drive)

    assert len(packet["candidates"]) == 1
    assert len(packet["cells"]) == 4
    assert len(packet["base_stress_comparisons"]) == 2
    assert len(packet["ohlc_olhc_comparisons"]) == 2
    assert packet["search_budget"]["attempts_remaining"] == 2
    assert packet["search_budget"]["proposal_slots_remaining"] == 13
    assert packet["cells"][0]["mtm_max_drawdown_fraction"] == {
        "status": "UNKNOWN",
        "value": None,
        "reason": "CONTINUOUS_MTM_EVIDENCE_INCOMPLETE",
    }
    serialized = json.dumps(packet, sort_keys=True)
    assert "ledger.jsonl" not in serialized
    assert "cells_path" not in serialized
    assert (
        "DRIVE_REMOTE_VERIFIED_IS_AN_EXTERNAL_RECEIPT_CLAIM_NOT_INDEPENDENT_PROOF"
        in packet["limitations"]
    )
    assert packet["live_permission"] is False
    assert verify_trainer_packet(packet) == packet


def test_partial_best_only_and_result_substitution_fail_closed(
    packet_inputs: dict,
) -> None:
    with pytest.raises(DojoAITrainerPacketError, match="partial or best-only"):
        _build(packet_inputs, cells=packet_inputs["cells"][:-1])

    substituted = copy.deepcopy(packet_inputs["evaluation"])
    substituted["study_sha256"] = "f" * 64
    body = {
        key: value for key, value in substituted.items() if key != "evaluation_sha256"
    }
    substituted["evaluation_sha256"] = _canonical_sha(body)
    with pytest.raises(DojoAITrainerPacketError, match="exact latest bound result"):
        _build(packet_inputs, evaluation=substituted)


def test_holdout_live_and_nonterminal_inputs_are_rejected(packet_inputs: dict) -> None:
    for field, value, match in (
        ("classification", "FORWARD_HOLDOUT", "holdout or forward"),
        ("live_permission", True, "research authority"),
        ("status", "RUNNING", "partial run"),
    ):
        run = copy.deepcopy(packet_inputs["run"])
        run[field] = value
        body = {key: item for key, item in run.items() if key != "run_sha256"}
        run["run_sha256"] = _canonical_sha(body)
        with pytest.raises(DojoAITrainerPacketError, match=match):
            _build(packet_inputs, run=run)


def test_drive_refs_reject_ledgers_paths_and_unverified_content(
    packet_inputs: dict,
) -> None:
    base = {
        "artifact_kind": "LEDGER",
        "drive_file_id": "driveFile123",
        "content_sha256": "6" * 64,
        "content_size_bytes": 321,
        "remote_verified": True,
        "metadata_receipt_sha256": "7" * 64,
    }
    with pytest.raises(DojoAITrainerPacketError, match="only manifests or reports"):
        _build(packet_inputs, drive_evidence_refs=[base])

    extra_path = {**base, "artifact_kind": "REPORT", "path": "/tmp/report"}
    with pytest.raises(DojoAITrainerPacketError, match="schema mismatch"):
        _build(packet_inputs, drive_evidence_refs=[extra_path])

    unverified = {**base, "artifact_kind": "MANIFEST", "remote_verified": False}
    with pytest.raises(DojoAITrainerPacketError, match="not remotely verified"):
        _build(packet_inputs, drive_evidence_refs=[unverified])


def test_packet_tamper_and_nan_are_rejected(packet_inputs: dict) -> None:
    packet = _build(packet_inputs)
    tampered = copy.deepcopy(packet)
    tampered["cells"][0]["terminal_net_jpy"] = 1.0
    with pytest.raises(DojoAITrainerPacketError, match="SHA-256 mismatch"):
        verify_trainer_packet(tampered)

    cell = copy.deepcopy(packet_inputs["cells"][0])
    cell["metrics"]["terminal_net_jpy"] = float("nan")
    cell["cell_sha256"] = "0" * 64
    cells = [cell, *packet_inputs["cells"][1:]]
    with pytest.raises(DojoAITrainerPacketError, match="non-finite"):
        _build(packet_inputs, cells=cells)


def test_rehashed_success_cell_metric_injection_cannot_replace_bound_truth(
    tmp_path: Path,
) -> None:
    sealed = _sealed_study()
    original_cells = _successful_cells(sealed)
    evaluation = _evaluation_from_cells(sealed, original_cells)
    inputs = _bind_inputs(
        tmp_path,
        sealed=sealed,
        evaluation=evaluation,
        cells=original_cells,
    )
    assert _build(inputs)["current_run"]["status"] == "COMPLETE"

    cells = copy.deepcopy(original_cells)
    attacked = next(
        cell
        for cell in cells
        if cell["intrabar"] == "OHLC" and cell["cost_arm"] == "STRESS"
    )
    attacked["metrics"].update(
        {
            "terminal_net_jpy": -5_000.0,
            "pair_pnl_jpy": {pair: -1_250.0 for pair in PAIRS},
            "mtm_max_drawdown_fraction": 0.20,
            "peak_margin_usage_fraction": 0.90,
            "leave_one_pair_out_net_jpy": {pair: -2_000.0 for pair in PAIRS},
        }
    )
    attacked_body = {
        key: value for key, value in attacked.items() if key != "cell_sha256"
    }
    attacked["cell_sha256"] = _canonical_sha(attacked_body)

    run = copy.deepcopy(inputs["run"])
    coordinate = next(
        row
        for row in run["coordinates"]
        if row["intrabar"] == "OHLC" and row["cost_arm"] == "STRESS"
    )
    coordinate["cell_sha256"] = attacked["cell_sha256"]
    for lopo in coordinate["lopo"]:
        lopo["terminal_net_jpy"] = -2_000.0
    run_body = {key: value for key, value in run.items() if key != "run_sha256"}
    run["run_sha256"] = _canonical_sha(run_body)

    with pytest.raises(
        DojoAITrainerPacketError,
        match="cells do not reconstruct the exact lineage-bound evaluation",
    ):
        _build(inputs, run=run, cells=cells)


def test_packet_seal_is_canonical_and_authority_free(packet_inputs: dict) -> None:
    packet = _build(packet_inputs)
    body = {key: value for key, value in packet.items() if key != "packet_sha256"}
    assert packet["packet_sha256"] == canonical_packet_sha256(body)
    assert packet["source_bindings"]["exact_result_binding_verified"] is True
    assert packet["proof_eligible"] is False
    assert packet["promotion_eligible"] is False
    assert packet["broker_mutation_allowed"] is False
    assert packet["order_authority"] == "NONE"
