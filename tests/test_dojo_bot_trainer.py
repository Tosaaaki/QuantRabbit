from __future__ import annotations

import hashlib
import json
import gzip
import os
import subprocess
import sys
from pathlib import Path

import pytest

import quant_rabbit.dojo_bot_trainer as trainer_module
from quant_rabbit.dojo_bot_catalog import CATALOG_CONTRACT, catalog_manifest
from quant_rabbit.dojo_bot_trainer import (
    CELL_CONTRACT,
    MAX_CANDIDATES,
    PROPOSAL_CONTRACT,
    STUDY_CONTRACT,
    DojoBotTrainerError,
    evaluate_training,
    score_ledger_metrics,
    seal_candidate_proposal,
    seal_cell_result,
    seal_study,
    verify_sealed_study,
)


PAIRS = ["CAD_JPY", "EUR_USD", "GBP_USD", "USD_JPY"]
SOURCES = {
    "bots/lab_bot.py": "a" * 64,
    "src/quant_rabbit/dojo_bot_trainer.py": "b" * 64,
}


def _bot_config(*, pairs: list[str] | None = None) -> dict:
    configured_pairs = pairs if pairs is not None else PAIRS
    return {
        "signal": "spike_fade",
        "pairs": configured_pairs,
        "tp_atr": 3.0,
        "sl_pips": 25.0,
        "ceiling_min": 60,
        "max_concurrent_per_pair": 1,
        "global_max_concurrent": len(configured_pairs),
        "per_pos_lev": 5.0,
        "atr_floor_pips": 0.5,
    }


def _proposal(candidate_id: str = "C1") -> dict:
    return seal_candidate_proposal(
        {
            "contract": PROPOSAL_CONTRACT,
            "schema_version": 1,
            "candidate_id": candidate_id,
            "family": "spike_fade",
            "hypothesis": "Release stale capital without increasing risk.",
            "config": _bot_config(),
            "risk_increase": False,
        }
    )


def _study(*, candidates: list[dict] | None = None) -> dict:
    return {
        "contract": STUDY_CONTRACT,
        "schema_version": 1,
        "study_id": "capital_rotation_train_v1",
        "window_role": "TRAIN",
        "initial_balance_jpy": 200_000.0,
        "trade_pairs": PAIRS,
        "feed_pairs": PAIRS,
        "candidates": candidates or [_proposal()],
        "window": {
            "start_utc": "2025-03-01T00:00:00Z",
            "end_utc": "2025-04-01T00:00:00Z",
            "corpus_id": "worn_fx_march_2025_v1",
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
            "model_claim": "gpt-5.6-sol",
            "provider_attestation": "UNVERIFIED",
        },
        "search_budget": {
            "attempt_ordinal": 1,
            "total_attempts_in_lineage": 5,
            "max_candidates": 4,
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


def _sealed_study(*, candidates: list[dict] | None = None) -> dict:
    return seal_study(_study(candidates=candidates), SOURCES)


def _cell_body(
    sealed: dict,
    *,
    artifact_root: Path,
    candidate_id: str = "C1",
    intrabar: str,
    cost_arm: str,
    terminal_net_jpy: float,
    mtm_complete: bool = True,
    lopo_replay_complete: bool = True,
) -> dict:
    proposal = {row["candidate_id"]: row for row in sealed["study"]["candidates"]}[
        candidate_id
    ]
    pair_value = terminal_net_jpy / len(PAIRS)
    coordinate = f"{candidate_id}-{intrabar}-{cost_arm}"
    main_payload = {
        "terminal_net_jpy": terminal_net_jpy,
        "cost_arm": cost_arm,
        "mtm_complete": mtm_complete,
        "pair_pnl_jpy": {pair: pair_value for pair in PAIRS},
    }

    def write_receipt(name: str, payload: dict, expected_pairs: list[str]) -> dict:
        relative = Path("sessions") / coordinate / name / "ledger.jsonl"
        target = artifact_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n"
        target.write_bytes(raw)
        score = _fake_artifact_score(raw, expected_pairs)
        return {
            "artifact_relpath": relative.as_posix(),
            "artifact_size_bytes": len(raw),
            "artifact_sha256": hashlib.sha256(raw).hexdigest(),
            "ledger_terminal_sha256": score["ledger_terminal_sha256"],
            "metrics_sha256": score["metrics_sha256"],
            "corpus_sha256": "1" * 64,
        }

    main_receipt = write_receipt("main", main_payload, PAIRS)
    lopo_receipts = {
        pair: write_receipt(
            f"lopo-{pair}",
            {
                "terminal_net_jpy": terminal_net_jpy * 0.60,
                "cost_arm": cost_arm,
                "mtm_complete": False,
                "pair_pnl_jpy": {
                    item: terminal_net_jpy * 0.60 / (len(PAIRS) - 1)
                    for item in PAIRS
                    if item != pair
                },
            },
            [item for item in PAIRS if item != pair],
        )
        for pair in PAIRS
    }
    return {
        "contract": CELL_CONTRACT,
        "schema_version": 1,
        "study_sha256": sealed["study_sha256"],
        "candidate_id": candidate_id,
        "proposal_sha256": proposal["proposal_sha256"],
        "intrabar": intrabar,
        "cost_arm": cost_arm,
        "execution_status": "SUCCESS",
        "failure_code": None,
        "metrics": {
            "terminal_net_jpy": terminal_net_jpy,
            "terminal_flat": True,
            "margin_closeouts": 0,
            "realized_max_drawdown_fraction": 0.04 if cost_arm == "BASE" else 0.07,
            "mtm_complete": mtm_complete,
            "mtm_max_drawdown_fraction": (0.06 if cost_arm == "BASE" else 0.09)
            if mtm_complete
            else None,
            "peak_margin_usage_fraction": 0.30 if cost_arm == "BASE" else 0.40,
            "fill_count": 20,
            "margin_reject_count": 1,
            "capital_lock_margin_jpy_hours": 100_000.0,
            "pair_pnl_jpy": {pair: pair_value for pair in PAIRS},
            "leave_one_pair_out_net_jpy": {
                pair: terminal_net_jpy * 0.60 for pair in PAIRS
            },
            "lopo_replay_complete": lopo_replay_complete,
        },
        "ledger_evidence": {
            "main": main_receipt,
            "lopo_by_pair": lopo_receipts,
        },
    }


def _fake_artifact_score(raw: bytes, expected_pairs: list[str]) -> dict:
    payload = json.loads(raw)
    net = float(payload["terminal_net_jpy"])
    mtm_complete = bool(payload["mtm_complete"])
    cost_arm = str(payload["cost_arm"])
    body = {
        "terminal_net_jpy": net,
        "terminal_flat": True,
        "margin_closeouts": 0,
        "realized_max_drawdown_fraction": 0.04 if cost_arm == "BASE" else 0.07,
        "mtm_complete": mtm_complete,
        "mtm_max_drawdown_fraction": (0.06 if cost_arm == "BASE" else 0.09)
        if mtm_complete
        else None,
        "peak_entry_margin_estimate_fraction": 0.30 if cost_arm == "BASE" else 0.40,
        "fill_count": 20,
        "margin_reject_count": 1,
        "capital_lock_margin_jpy_hours": 100_000.0,
        "pair_pnl_jpy": {
            pair: float(payload["pair_pnl_jpy"][pair]) for pair in expected_pairs
        },
        "ledger_size_bytes": len(raw),
        "ledger_file_sha256": hashlib.sha256(raw).hexdigest(),
        "ledger_terminal_sha256": hashlib.sha256(b"terminal:" + raw).hexdigest(),
        "corpus_sha256": "1" * 64,
    }
    return {**body, "metrics_sha256": _canonical_sha(body)}


@pytest.fixture
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    def fake_score(
        ledger,
        _start_balance,
        expected_pairs,
        _window_start,
        _window_end,
        **_kwargs,
    ):
        raw = ledger.read()
        return _fake_artifact_score(raw, list(expected_pairs))

    monkeypatch.setattr(trainer_module, "score_ledger_metrics", fake_score)
    return tmp_path


def _cells(
    sealed: dict,
    artifact_root: Path,
    *,
    candidate_id: str = "C1",
    mtm_complete: bool = True,
    lopo_replay_complete: bool = True,
) -> list[dict]:
    net = {
        ("OHLC", "BASE"): 10_000.0,
        ("OHLC", "STRESS"): 6_000.0,
        ("OLHC", "BASE"): 8_000.0,
        ("OLHC", "STRESS"): 5_000.0,
    }
    return [
        seal_cell_result(
            _cell_body(
                sealed,
                artifact_root=artifact_root,
                candidate_id=candidate_id,
                intrabar=path,
                cost_arm=cost,
                terminal_net_jpy=net[(path, cost)],
                mtm_complete=mtm_complete,
                lopo_replay_complete=lopo_replay_complete,
            ),
            sealed,
            artifact_root=artifact_root,
        )
        for path in ("OHLC", "OLHC")
        for cost in ("BASE", "STRESS")
    ]


def test_study_round_trip_binds_sources_and_safety_boundary() -> None:
    sealed = _sealed_study()

    assert verify_sealed_study(sealed, SOURCES) == sealed
    assert sealed["classification"] == "WORN_HISTORICAL_TRAIN_ONLY"
    assert sealed["proof_eligible"] is False
    assert sealed["promotion_eligible"] is False
    assert sealed["live_permission"] is False
    assert sealed["order_authority"] == "NONE"
    assert sealed["broker_mutation_allowed"] is False
    assert sealed["study"]["window"] == {
        "start_utc": "2025-03-01T00:00:00+00:00",
        "end_utc": "2025-04-01T00:00:00+00:00",
        "corpus_id": "worn_fx_march_2025_v1",
        "corpus_sha256": "1" * 64,
        "evidence_tier": "WORN_TRAIN",
    }
    assert sealed["study"]["cost_arms"] == {
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
    }
    assert sealed["study"]["proposer_evidence"] == {
        "prompt_sha256": "2" * 64,
        "input_sha256": "3" * 64,
        "raw_response_sha256": "4" * 64,
        "model_claim": "gpt-5.6-sol",
        "provider_attestation": "UNVERIFIED",
    }
    assert sealed["study"]["search_budget"] == {
        "attempt_ordinal": 1,
        "total_attempts_in_lineage": 5,
        "max_candidates": 4,
    }

    with pytest.raises(DojoBotTrainerError, match="source digest drift"):
        verify_sealed_study(sealed, {**SOURCES, "bots/candidate.py": "c" * 64})


def test_study_seal_binds_window_cost_proposer_and_search_metadata() -> None:
    sealed = _sealed_study()
    for path, replacement in (
        (("window", "corpus_id"), "different_corpus"),
        (("cost_arms", "STRESS", "recorded_spread_multiplier"), 2.0),
        (("proposer_evidence", "model_claim"), "different-model"),
        (("search_budget", "total_attempts_in_lineage"), 6),
    ):
        tampered = json.loads(json.dumps(sealed))
        target = tampered["study"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        with pytest.raises(DojoBotTrainerError, match="digest mismatch|must equal 1"):
            verify_sealed_study(tampered, SOURCES)


def test_window_cost_proposer_and_search_contracts_fail_closed() -> None:
    reversed_window = _study()
    reversed_window["window"]["end_utc"] = reversed_window["window"]["start_utc"]
    with pytest.raises(DojoBotTrainerError, match="start_utc < end_utc"):
        seal_study(reversed_window, SOURCES)

    wrong_tier = _study()
    wrong_tier["window"]["evidence_tier"] = "HOLDOUT"
    with pytest.raises(DojoBotTrainerError, match="WORN_TRAIN"):
        seal_study(wrong_tier, SOURCES)

    extra_cost_arm = _study()
    extra_cost_arm["cost_arms"]["OPTIMISTIC"] = dict(
        extra_cost_arm["cost_arms"]["BASE"]
    )
    with pytest.raises(DojoBotTrainerError, match="schema mismatch"):
        seal_study(extra_cost_arm, SOURCES)

    base_multiplier = _study()
    base_multiplier["cost_arms"]["BASE"]["recorded_spread_multiplier"] = 0.9
    with pytest.raises(DojoBotTrainerError, match="BASE recorded spread"):
        seal_study(base_multiplier, SOURCES)

    cheaper_stress = _study()
    cheaper_stress["cost_arms"]["BASE"]["financing_pips_per_day"] = 0.9
    with pytest.raises(DojoBotTrainerError, match="lower than BASE"):
        seal_study(cheaper_stress, SOURCES)

    identical_stress = _study()
    identical_stress["cost_arms"]["STRESS"] = dict(
        identical_stress["cost_arms"]["BASE"]
    )
    with pytest.raises(DojoBotTrainerError, match="strictly harsher"):
        seal_study(identical_stress, SOURCES)

    unsupported_attestation = _study()
    unsupported_attestation["proposer_evidence"]["provider_attestation"] = "CLAIMED"
    with pytest.raises(DojoBotTrainerError, match="attestation"):
        seal_study(unsupported_attestation, SOURCES)

    ordinal_overrun = _study()
    ordinal_overrun["search_budget"]["attempt_ordinal"] = 6
    with pytest.raises(DojoBotTrainerError, match="ordinal exceeds"):
        seal_study(ordinal_overrun, SOURCES)

    candidates_over_budget = _study(candidates=[_proposal("C1"), _proposal("C2")])
    candidates_over_budget["search_budget"]["max_candidates"] = 1
    with pytest.raises(DojoBotTrainerError, match="sealed search budget"):
        seal_study(candidates_over_budget, SOURCES)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("normal_mtm_drawdown_max", 0.100001),
        ("stress_mtm_drawdown_max", 0.150001),
        ("peak_margin_usage_max", 0.450001),
        ("margin_reject_rate_max", 0.100001),
        ("cost_retention_min", 0.499999),
        ("pair_positive_share_max", 0.500001),
        ("pair_hhi_max", 0.500001),
    ],
)
def test_repo_owned_threshold_policy_cannot_be_loosened(
    field: str, value: float
) -> None:
    study = _study()
    study["thresholds"][field] = value

    with pytest.raises(DojoBotTrainerError, match="repo-owned hard policy"):
        seal_study(study, SOURCES)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("slippage_pips_per_fill", 0.299999),
        ("financing_pips_per_day", 0.799999),
    ],
)
def test_repo_owned_stress_cost_floor_cannot_be_weakened(
    field: str, value: float
) -> None:
    study = _study()
    study["cost_arms"]["STRESS"][field] = value

    with pytest.raises(DojoBotTrainerError, match="repo-owned hard floor"):
        seal_study(study, SOURCES)


def test_train_window_shorter_than_28_calendar_days_is_rejected() -> None:
    study = _study()
    study["window"]["end_utc"] = "2025-03-28T00:00:00Z"

    with pytest.raises(DojoBotTrainerError, match="at least 28 days"):
        seal_study(study, SOURCES)


def test_strict_json_duplicate_nan_unknown_schema_and_candidate_cap_fail_closed() -> (
    None
):
    with pytest.raises(DojoBotTrainerError, match="duplicate JSON key"):
        verify_sealed_study(
            '{"contract":"x","contract":"y"}',
            SOURCES,
        )

    nan_study = _study()
    nan_study["initial_balance_jpy"] = float("nan")
    with pytest.raises(DojoBotTrainerError, match="non-finite"):
        seal_study(nan_study, SOURCES)

    unknown = _study()
    unknown["contract"] = "QR_DOJO_BOT_TRAINER_STUDY_V999"
    with pytest.raises(DojoBotTrainerError, match="unsupported"):
        seal_study(unknown, SOURCES)

    candidates = [_proposal(f"C{index:02d}") for index in range(MAX_CANDIDATES + 1)]
    with pytest.raises(DojoBotTrainerError, match="candidates"):
        seal_study(_study(candidates=candidates), SOURCES)


def test_candidate_proposal_tamper_and_risk_increase_fail_closed() -> None:
    tampered = _proposal()
    tampered["config"]["ceiling_min"] = 30
    with pytest.raises(DojoBotTrainerError, match="catalog binding drift"):
        seal_study(_study(candidates=[tampered]), SOURCES)

    risky = {
        "contract": PROPOSAL_CONTRACT,
        "schema_version": 1,
        "candidate_id": "RISKY",
        "family": "spike_fade",
        "hypothesis": "Increase leverage.",
        "config": _bot_config(),
        "risk_increase": True,
    }
    with pytest.raises(DojoBotTrainerError, match="may not increase risk"):
        seal_candidate_proposal(risky)


def test_candidate_proposal_binds_reviewed_catalog_and_exact_study_pairs() -> None:
    proposal = _proposal()

    assert proposal["family"] == proposal["config"]["signal"] == "spike_fade"
    assert proposal["config"]["pairs"] == PAIRS
    assert proposal["catalog_contract"] == CATALOG_CONTRACT
    assert proposal["catalog_sha256"] == _canonical_sha(catalog_manifest())
    assert proposal["config_sha256"] == _canonical_sha(proposal["config"])

    mismatched_family = {
        "contract": PROPOSAL_CONTRACT,
        "schema_version": 1,
        "candidate_id": "MISMATCH",
        "family": "burst",
        "hypothesis": "Mismatch must fail closed.",
        "config": _bot_config(),
        "risk_increase": False,
    }
    with pytest.raises(DojoBotTrainerError, match="family does not match"):
        seal_candidate_proposal(mismatched_family)

    subset = seal_candidate_proposal(
        {
            "contract": PROPOSAL_CONTRACT,
            "schema_version": 1,
            "candidate_id": "SUBSET",
            "family": "spike_fade",
            "hypothesis": "A different pair universe must not enter this study.",
            "config": _bot_config(pairs=PAIRS[:2]),
            "risk_increase": False,
        }
    )
    with pytest.raises(DojoBotTrainerError, match="trade_pairs exactly"):
        seal_study(_study(candidates=[subset]), SOURCES)

    missing_trade_feed = _study()
    missing_trade_feed["feed_pairs"] = ["AUD_USD", *PAIRS[:-1]]
    with pytest.raises(DojoBotTrainerError, match="include every trade pair"):
        seal_study(missing_trade_feed, SOURCES)


def test_complete_grid_uses_coordinate_worst_and_gate_first_ranking(
    artifact_root: Path,
) -> None:
    sealed = _sealed_study()
    result = evaluate_training(
        sealed,
        _cells(sealed, artifact_root),
        artifact_root=artifact_root,
    )
    candidate = result["candidate_evaluations"][0]

    assert result["fixed_denominator"] == {
        "candidate_count": 1,
        "intrabar_paths": ["OHLC", "OLHC"],
        "cost_arms": ["BASE", "STRESS"],
        "expected_cell_count": 4,
        "observed_cell_count": 4,
        "coordinate_receipts_complete": True,
        "execution_success_complete": True,
    }
    assert candidate["status"] == "TRAIN_DIAGNOSTIC_PASS"
    assert candidate["diagnostic_rank_eligible"] is True
    assert candidate["coordinate_worst"]["terminal_net_jpy"] == 5_000.0
    assert candidate["coordinate_worst"]["normal_effective_drawdown_fraction"] == 0.06
    assert candidate["coordinate_worst"]["stress_effective_drawdown_fraction"] == 0.09
    assert candidate["coordinate_worst"]["peak_margin_usage_fraction"] == 0.40
    assert candidate["coordinate_worst"]["cost_retention"] == 0.60
    assert candidate["coordinate_worst"]["pair_positive_share"] == 0.25
    assert candidate["coordinate_worst"]["pair_hhi"] == 0.25
    assert 0 < candidate["diagnostic_score"] <= 100
    assert candidate["risk_policy_receipt"]["rankable"] is True
    assert candidate["proposer_risk_claim_ignored"] is True
    assert result["diagnostic_ranking"] == ["C1"]
    assert result["rank_eligible_candidate_ids"] == ["C1"]
    assert result["unranked_candidate_ids"] == []
    assert result["promotion_eligible"] is False
    assert result["live_permission"] is False
    assert result["order_authority"] == "NONE"


def test_missing_duplicate_and_tampered_cells_fail_closed(
    artifact_root: Path,
) -> None:
    sealed = _sealed_study()
    cells = _cells(sealed, artifact_root)

    with pytest.raises(DojoBotTrainerError, match="fixed denominator"):
        evaluate_training(sealed, cells[:-1], artifact_root=artifact_root)

    duplicate = [*cells[:-1], cells[0]]
    with pytest.raises(DojoBotTrainerError, match="duplicate"):
        evaluate_training(sealed, duplicate, artifact_root=artifact_root)

    tampered = json.loads(json.dumps(cells))
    tampered[0]["metrics"]["peak_margin_usage_fraction"] += 0.01
    with pytest.raises(DojoBotTrainerError, match="seal mismatch"):
        evaluate_training(sealed, tampered, artifact_root=artifact_root)

    tampered_evidence = json.loads(json.dumps(cells))
    tampered_evidence[0]["ledger_evidence"]["main"]["metrics_sha256"] = "f" * 64
    with pytest.raises(DojoBotTrainerError, match="seal mismatch"):
        evaluate_training(sealed, tampered_evidence, artifact_root=artifact_root)


def test_cell_ledger_evidence_requires_main_and_exact_lopo_pair_ledgers(
    artifact_root: Path,
) -> None:
    sealed = _sealed_study()
    body = _cell_body(
        sealed,
        artifact_root=artifact_root,
        intrabar="OHLC",
        cost_arm="BASE",
        terminal_net_jpy=10_000.0,
    )
    del body["ledger_evidence"]["lopo_by_pair"][PAIRS[0]]
    with pytest.raises(DojoBotTrainerError, match="every expected pair exactly"):
        seal_cell_result(body, sealed, artifact_root=artifact_root)

    body = _cell_body(
        sealed,
        artifact_root=artifact_root,
        intrabar="OHLC",
        cost_arm="BASE",
        terminal_net_jpy=10_000.0,
    )
    body["ledger_evidence"]["main"]["metrics_sha256"] = "A" * 64
    with pytest.raises(DojoBotTrainerError, match="lowercase SHA-256"):
        seal_cell_result(body, sealed, artifact_root=artifact_root)


def test_failed_runner_cell_is_sealed_retained_and_rejected(
    artifact_root: Path,
) -> None:
    sealed = _sealed_study()
    cells = _cells(sealed, artifact_root)
    body = _cell_body(
        sealed,
        artifact_root=artifact_root,
        intrabar="OHLC",
        cost_arm="BASE",
        terminal_net_jpy=0.0,
        mtm_complete=False,
        lopo_replay_complete=False,
    )
    body["execution_status"] = "FAILED"
    body["failure_code"] = "RUNNER_PROCESS_EXIT_NONZERO"
    body["metrics"] = {
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
        "pair_pnl_jpy": {pair: 0.0 for pair in PAIRS},
        "leave_one_pair_out_net_jpy": {pair: 0.0 for pair in PAIRS},
        "lopo_replay_complete": False,
    }
    body["ledger_evidence"] = {
        "main": {
            "artifact_relpath": None,
            "artifact_size_bytes": 0,
            "artifact_sha256": "0" * 64,
            "ledger_terminal_sha256": "0" * 64,
            "metrics_sha256": "0" * 64,
            "corpus_sha256": "0" * 64,
        },
        "lopo_by_pair": {
            pair: {
                "artifact_relpath": None,
                "artifact_size_bytes": 0,
                "artifact_sha256": "0" * 64,
                "ledger_terminal_sha256": "0" * 64,
                "metrics_sha256": "0" * 64,
                "corpus_sha256": "0" * 64,
            }
            for pair in PAIRS
        },
    }
    cells[0] = seal_cell_result(body, sealed, artifact_root=artifact_root)

    result = evaluate_training(sealed, cells, artifact_root=artifact_root)
    candidate = result["candidate_evaluations"][0]

    assert result["fixed_denominator"]["observed_cell_count"] == 4
    assert result["fixed_denominator"]["coordinate_receipts_complete"] is True
    assert result["fixed_denominator"]["execution_success_complete"] is False
    assert candidate["status"] == "TRAIN_REJECT"
    assert candidate["diagnostic_rank_eligible"] is False
    assert candidate["diagnostic_score"] is None
    assert "RUNNER_CELL_FAILURE" in candidate["gate_blockers"]
    assert candidate["failed_coordinates"] == ["OHLC:BASE:RUNNER_PROCESS_EXIT_NONZERO"]


def test_mtm_incomplete_fails_diagnostic_ranking_and_never_promotes(
    artifact_root: Path,
) -> None:
    sealed = _sealed_study()
    result = evaluate_training(
        sealed,
        _cells(sealed, artifact_root, mtm_complete=False),
        artifact_root=artifact_root,
    )
    candidate = result["candidate_evaluations"][0]

    assert candidate["status"] == "TRAIN_REJECT"
    assert candidate["diagnostic_rank_eligible"] is False
    assert candidate["diagnostic_score"] is None
    assert candidate["mtm_complete"] is False
    assert (
        candidate["mtm_incomplete_uses_realized_dd_for_train_diagnostic_only"] is False
    )
    assert "CONTINUOUS_MTM_EVIDENCE_INCOMPLETE" in candidate["gate_blockers"]
    assert "CONTINUOUS_MTM_EVIDENCE_INCOMPLETE" in candidate["promotion_blockers"]
    assert candidate["promotion_gate_passed"] is False
    assert candidate["promotion_eligible"] is False
    assert result["rank_eligible_candidate_ids"] == []
    assert result["diagnostic_ranking"] == []
    assert result["unranked_candidate_ids"] == ["C1"]


def test_caller_cannot_claim_incomplete_lopo_for_successful_receipts(
    artifact_root: Path,
) -> None:
    sealed = _sealed_study()
    body = _cell_body(
        sealed,
        artifact_root=artifact_root,
        intrabar="OHLC",
        cost_arm="BASE",
        terminal_net_jpy=10_000.0,
        lopo_replay_complete=False,
    )
    with pytest.raises(DojoBotTrainerError, match="do not match metrics rebuilt"):
        seal_cell_result(body, sealed, artifact_root=artifact_root)


def test_caller_supplied_metrics_cannot_be_resealed(
    artifact_root: Path,
) -> None:
    sealed = _sealed_study()
    cells = _cells(sealed, artifact_root)
    body = {key: value for key, value in cells[0].items() if key != "cell_sha256"}
    body["metrics"]["margin_reject_count"] = 20

    with pytest.raises(DojoBotTrainerError, match="do not match metrics rebuilt"):
        seal_cell_result(body, sealed, artifact_root=artifact_root)


def test_evaluation_reopens_artifact_and_rejects_post_seal_mutation(
    artifact_root: Path,
) -> None:
    sealed = _sealed_study()
    cells = _cells(sealed, artifact_root)
    receipt = cells[0]["ledger_evidence"]["main"]
    path = artifact_root / receipt["artifact_relpath"]
    payload = json.loads(path.read_bytes())
    payload["terminal_net_jpy"] = 999_999.0
    path.write_bytes(json.dumps(payload, sort_keys=True).encode() + b"\n")

    with pytest.raises(DojoBotTrainerError, match="artifact_size_bytes"):
        evaluate_training(sealed, cells, artifact_root=artifact_root)


def test_receipt_paths_reject_traversal_and_symlinks(artifact_root: Path) -> None:
    sealed = _sealed_study()
    body = _cell_body(
        sealed,
        artifact_root=artifact_root,
        intrabar="OHLC",
        cost_arm="BASE",
        terminal_net_jpy=10_000.0,
    )
    body["ledger_evidence"]["main"]["artifact_relpath"] = "../ledger.jsonl"
    with pytest.raises(DojoBotTrainerError, match="safe canonical ledger path"):
        seal_cell_result(body, sealed, artifact_root=artifact_root)

    body = _cell_body(
        sealed,
        artifact_root=artifact_root,
        intrabar="OHLC",
        cost_arm="BASE",
        terminal_net_jpy=10_000.0,
    )
    receipt = body["ledger_evidence"]["main"]
    target = artifact_root / receipt["artifact_relpath"]
    replacement = artifact_root / "replacement-ledger.jsonl"
    replacement.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(replacement)
    with pytest.raises(DojoBotTrainerError, match="regular non-symlink"):
        seal_cell_result(body, sealed, artifact_root=artifact_root)


def test_artifact_mutation_during_scoring_is_detected(
    artifact_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed = _sealed_study()
    body = _cell_body(
        sealed,
        artifact_root=artifact_root,
        intrabar="OHLC",
        cost_arm="BASE",
        terminal_net_jpy=10_000.0,
    )
    receipt = body["ledger_evidence"]["main"]
    target = artifact_root / receipt["artifact_relpath"]
    original_score = trainer_module.score_ledger_metrics
    mutated = False

    def mutating_score(*args, **kwargs):
        nonlocal mutated
        result = original_score(*args, **kwargs)
        if not mutated:
            mutated = True
            target.write_bytes(target.read_bytes() + b" ")
        return result

    monkeypatch.setattr(trainer_module, "score_ledger_metrics", mutating_score)
    with pytest.raises(DojoBotTrainerError, match="changed during verification"):
        seal_cell_result(body, sealed, artifact_root=artifact_root)


def test_cell_seal_and_evaluation_require_trusted_artifact_root() -> None:
    sealed = _sealed_study()
    with pytest.raises(TypeError, match="artifact_root"):
        seal_cell_result({}, sealed)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="artifact_root"):
        evaluate_training(sealed, [])  # type: ignore[call-arg]


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


LEDGER_OWNER = "dojo:test-bot"
LEDGER_MODULE_SHA = "5" * 64
LEDGER_CONFIG_SHA = "6" * 64
LEDGER_DEPENDENCIES = {
    "bots/lab_bot.py": "7" * 64,
    "src/quant_rabbit/dojo_bot_catalog.py": "8" * 64,
}


def _write_ledger(path: Path) -> dict[str, str]:
    start = "2025-03-03T00:00:00Z"
    end = "2025-03-04T00:00:00Z"
    repository_root = Path(__file__).resolve().parents[1]
    corpus_body = {
        "root": "/fixture/corpus",
        "shards": [
            {
                "path": "oanda/USD_JPY/USD_JPY_M1.fixture.jsonl.gz",
                "size_bytes": 123,
                "sha256": "9" * 64,
            }
        ],
    }
    corpus = {**corpus_body, "corpus_sha256": _canonical_sha(corpus_body)}
    manifest_body = {
        "schema": "QR_VIRTUAL_SESSION_REPRODUCIBILITY_V1",
        "source": {
            "git_head": "test-head",
            "session_script_sha256": hashlib.sha256(
                (
                    repository_root / "scripts" / "run-virtual-market-session.py"
                ).read_bytes()
            ).hexdigest(),
            "virtual_broker_sha256": hashlib.sha256(
                (repository_root / "src/quant_rabbit/virtual_broker.py").read_bytes()
            ).hexdigest(),
            "python_executable": "/python",
            "python_version": "test",
        },
        "replay": {
            "feed": "replay",
            "time_from": start,
            "time_to": end,
            "pairs": ["CAD_JPY", "USD_JPY"],
            "granularity": "M1",
            "intrabar": "OHLC",
            "bot_bar": "feed",
            "period_end_settlement": True,
        },
        "corpus": corpus,
        "costs": {
            "slippage_pips_per_fill": 0.2,
            "financing_pips_per_day": 0.05,
            "leverage": 25.0,
        },
        "initial_balance_jpy": 1_000.0,
        "resume_snapshot": None,
        "bot": {
            "kind": "custom_module",
            "name": None,
            "module_path": "/fixture/bots/lab_bot.py",
            "module_sha256": LEDGER_MODULE_SHA,
            "class": "Bot",
            "strategy_owner_id": LEDGER_OWNER,
            "dependency_sha256": LEDGER_DEPENDENCIES,
            "configuration_bindings": {
                "DOJO_BOT_CONFIG": {
                    "sha256": LEDGER_CONFIG_SHA,
                    "length": 123,
                }
            },
        },
        "pacing": {"bars_per_second": 100_000.0, "step": False, "state_every": 1},
        "order_authority": "NONE",
    }
    manifest = {
        **manifest_body,
        "manifest_sha256": _canonical_sha(manifest_body),
    }
    rows = [
        (
            "SESSION_START",
            {
                "contract": "QR_VIRTUAL_MARKET_SESSION_V1",
                "feed": "replay",
                "balance": 1_000.0,
                "pairs": "CAD_JPY,USD_JPY",
                "order_authority": "NONE",
                "reproducibility_manifest": manifest,
                "reproducibility_manifest_sha256": manifest["manifest_sha256"],
            },
        ),
        (
            "BOT_LOADED",
            {
                "module": "/fixture/bots/lab_bot.py",
                "class": "Bot",
                "strategy_owner_id": LEDGER_OWNER,
            },
        ),
        (
            "FILL_LIMIT",
            {
                "trade_id": "T1",
                "pair": "USD_JPY",
                "units": 100.0,
                "price": 100.0,
                "slippage_pips": 0.2,
                "strategy_owner_id": LEDGER_OWNER,
                "conversion": {"rate_jpy_per_quote_unit": 1.0},
                "quote": {"bid": 99.9, "ask": 100.0, "ts": "2025-03-03T00:00:00Z"},
            },
        ),
        (
            "EXIT_SL",
            {
                "trade_id": "T1",
                "pl_jpy": -20.0,
                "slippage_pips": 0.2,
                "strategy_owner_id": LEDGER_OWNER,
                "quote": {"bid": 99.8, "ask": 99.9, "ts": "2025-03-03T00:30:00Z"},
            },
        ),
        (
            "FILL_LIMIT",
            {
                "trade_id": "T2",
                "pair": "CAD_JPY",
                "units": 100.0,
                "price": 80.0,
                "slippage_pips": 0.2,
                "strategy_owner_id": LEDGER_OWNER,
                "conversion": {"rate_jpy_per_quote_unit": 1.0},
                "quote": {"bid": 79.9, "ask": 80.0, "ts": "2025-03-03T01:00:00Z"},
            },
        ),
        (
            "LIMIT_REJECTED_INSUFFICIENT_MARGIN",
            {
                "order_id": "O3",
                "pair": "USD_JPY",
                "strategy_owner_id": LEDGER_OWNER,
            },
        ),
        (
            "EXIT_TP",
            {
                "trade_id": "T2",
                "pl_jpy": 30.0,
                "slippage_pips": 0.2,
                "strategy_owner_id": LEDGER_OWNER,
                "quote": {"bid": 80.3, "ask": 80.4, "ts": "2025-03-03T02:00:00Z"},
            },
        ),
        (
            "PERIOD_END_SETTLEMENT",
            {
                "strategy_owner_id": LEDGER_OWNER,
                "complete": True,
                "errors": [],
            },
        ),
        (
            "SESSION_STOP",
            {
                "account": {
                    "balance_jpy": 1_010.0,
                    "open_positions": 0,
                    "resting_orders": 0,
                }
            },
        ),
    ]
    previous = "0" * 64
    sealed = []
    for index, (event, payload) in enumerate(rows):
        body = {
            "ts_utc": f"2026-07-19T00:00:{index:02d}+00:00",
            "event": event,
            "payload": payload,
            "prev_sha": previous,
        }
        digest = _canonical_sha(body)
        sealed.append({**body, "sha": digest})
        previous = digest
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in sealed
        ),
        encoding="utf-8",
    )
    return {
        "corpus_sha256": corpus["corpus_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
    }


def _score_fixture_ledger(
    ledger: Path,
    evidence: dict[str, str],
    **overrides: object,
) -> dict:
    expected: dict[str, object] = {
        "expected_intrabar": "OHLC",
        "expected_slippage_pips_per_fill": 0.2,
        "expected_financing_pips_per_day": 0.05,
        "expected_corpus_sha256": evidence["corpus_sha256"],
        "expected_bot_config_sha256": LEDGER_CONFIG_SHA,
        "expected_strategy_owner_id": LEDGER_OWNER,
        "expected_bot_module_sha256": LEDGER_MODULE_SHA,
        "expected_bot_dependency_sha256": LEDGER_DEPENDENCIES,
    }
    expected.update(overrides)
    return score_ledger_metrics(
        ledger,
        1_000.0,
        ["CAD_JPY", "USD_JPY"],
        "2025-03-03T00:00:00Z",
        "2025-03-04T00:00:00Z",
        **expected,
    )


def _rewrite_ledger_chain(path: Path, rows: list[dict]) -> None:
    previous = "0" * 64
    sealed = []
    for row in rows:
        body = {
            "ts_utc": row["ts_utc"],
            "event": row["event"],
            "payload": row["payload"],
            "prev_sha": previous,
        }
        digest = _canonical_sha(body)
        sealed.append({**body, "sha": digest})
        previous = digest
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in sealed),
        encoding="utf-8",
    )


def test_score_ledger_metrics_rebuilds_realized_concentration_and_lock(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    evidence = _write_ledger(ledger)

    result = _score_fixture_ledger(ledger, evidence)

    assert result["terminal_net_jpy"] == 10.0
    assert result["terminal_flat"] is True
    assert result["fill_count"] == 2
    assert result["margin_reject_count"] == 1
    assert result["margin_reject_rate"] == pytest.approx(1 / 3)
    assert result["realized_max_drawdown_fraction"] == 0.02
    assert result["peak_entry_margin_estimate_jpy"] == 400.0
    assert result["peak_entry_margin_estimate_fraction"] == 0.40
    assert result["pair_pnl_jpy"] == {"CAD_JPY": 30.0, "USD_JPY": -20.0}
    assert result["positive_pair_max_share"] == 1.0
    assert result["positive_pair_hhi"] == 1.0
    assert result["effective_positive_pairs"] == 1.0
    assert result["leave_one_pair_out_net_jpy"] == {
        "CAD_JPY": -20.0,
        "USD_JPY": 30.0,
    }
    assert result["lopo_basis"] == "ADDITIVE_ATTRIBUTION_NOT_COUNTERFACTUAL_REPLAY"
    assert result["lopo_replay_complete"] is False
    assert result["capital_lock_margin_jpy_hours"] == 520.0
    assert result["unit_weighted_hold_hours"] == 0.75
    assert result["mtm_complete"] is False
    assert result["mtm_evidence_status"] == "NO_ACCOUNT_MARK_SEQUENCE"
    assert result["unverified_mtm_claim_present"] is False
    assert result["mtm_max_drawdown_fraction"] is None
    assert result["promotion_eligible"] is False
    assert result["live_permission"] is False
    assert result["order_authority"] == "NONE"
    assert result["reproducibility_manifest_sha256"] == evidence["manifest_sha256"]
    assert result["corpus_sha256"] == evidence["corpus_sha256"]
    assert result["bot_config_sha256"] == LEDGER_CONFIG_SHA
    assert result["bot_module_sha256"] == LEDGER_MODULE_SHA
    assert result["strategy_owner_id"] == LEDGER_OWNER


def test_score_digest_is_relocatable_between_path_and_verified_stream(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    evidence = _write_ledger(ledger)

    by_path = _score_fixture_ledger(ledger, evidence)
    with ledger.open("rb") as handle:
        by_stream = _score_fixture_ledger(
            handle,  # type: ignore[arg-type]
            evidence,
            ledger_artifact_path="sessions/example/ledger.jsonl",
        )

    assert by_path["ledger_path"] != by_stream["ledger_path"]
    assert by_path["ledger_file_sha256"] == by_stream["ledger_file_sha256"]
    assert by_path["metrics_sha256"] == by_stream["metrics_sha256"]


def test_score_ledger_metrics_rejects_hash_tamper(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    evidence = _write_ledger(ledger)
    rows = ledger.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(rows[2])
    tampered["payload"]["pl_jpy"] = 999.0
    rows[2] = json.dumps(tampered, sort_keys=True)
    ledger.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(DojoBotTrainerError, match="digest mismatch"):
        _score_fixture_ledger(ledger, evidence)


def test_score_ledger_metrics_does_not_trust_terminal_mtm_self_claim(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    evidence = _write_ledger(ledger)
    rows = [
        json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()
    ]
    rows.insert(
        -1,
        {
            "ts_utc": "2026-07-19T00:00:08+00:00",
            "event": "ACCOUNT_MARK",
            "payload": {"account": {"equity_jpy": 900.0, "margin_used_jpy": 0.0}},
        },
    )
    rows[-1]["payload"]["mtm_complete"] = True
    rows[-1]["payload"]["mtm_mark_count"] = 1
    _rewrite_ledger_chain(ledger, rows)

    result = _score_fixture_ledger(ledger, evidence)

    assert result["mtm_complete"] is False
    assert result["mtm_evidence_status"] == "UNVERIFIED_ACCOUNT_MARK_SEQUENCE"
    assert result["unverified_mtm_claim_present"] is True
    assert result["mtm_mark_count"] == 1
    assert result["mtm_max_drawdown_fraction"] is None


def test_score_ledger_metrics_rejects_wrong_replay_and_bot_bindings(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    evidence = _write_ledger(ledger)

    with pytest.raises(DojoBotTrainerError, match="mechanics mismatch"):
        _score_fixture_ledger(ledger, evidence, expected_intrabar="OLHC")
    with pytest.raises(DojoBotTrainerError, match="sealed study costs"):
        _score_fixture_ledger(
            ledger,
            evidence,
            expected_slippage_pips_per_fill=0.3,
        )
    with pytest.raises(DojoBotTrainerError, match="sealed study corpus"):
        _score_fixture_ledger(
            ledger,
            evidence,
            expected_corpus_sha256="f" * 64,
        )
    with pytest.raises(DojoBotTrainerError, match="strategy owner"):
        _score_fixture_ledger(
            ledger,
            evidence,
            expected_strategy_owner_id="dojo:different-owner",
        )
    with pytest.raises(DojoBotTrainerError, match="custom bot binding"):
        _score_fixture_ledger(
            ledger,
            evidence,
            expected_bot_dependency_sha256={"bots/lab_bot.py": "e" * 64},
        )


def test_score_ledger_metrics_verifies_manifest_self_digest(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    evidence = _write_ledger(ledger)
    rows = [
        json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["payload"]["reproducibility_manifest"]["costs"]["leverage"] = 20.0
    _rewrite_ledger_chain(ledger, rows)

    with pytest.raises(DojoBotTrainerError, match="manifest hash mismatch"):
        _score_fixture_ledger(ledger, evidence)

    ledger = tmp_path / "session-digest.jsonl"
    evidence = _write_ledger(ledger)
    rows = [
        json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()
    ]
    manifest = rows[0]["payload"]["reproducibility_manifest"]
    manifest["pacing"]["state_every"] = 2
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = _canonical_sha(body)
    _rewrite_ledger_chain(ledger, rows)

    with pytest.raises(DojoBotTrainerError, match="SESSION_START manifest digest"):
        _score_fixture_ledger(ledger, evidence)


def _write_mtm_corpus(root: Path) -> None:
    for pair_index, pair in enumerate(("CAD_JPY", "USD_JPY")):
        shard = root / "fixture" / pair / f"{pair}_M1_BA_2026.jsonl.gz"
        shard.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(shard, "wt", encoding="utf-8") as handle:
            for minute in range(2):
                base = 80.0 + pair_index * 70.0 + minute * 0.01
                handle.write(
                    json.dumps(
                        {
                            "time": f"2026-01-01T00:0{minute}:00Z",
                            "bid": {
                                "o": base,
                                "h": base + 0.02,
                                "l": base - 0.02,
                                "c": base + 0.01,
                            },
                            "ask": {
                                "o": base + 0.01,
                                "h": base + 0.03,
                                "l": base - 0.01,
                                "c": base + 0.02,
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )


def _strict_mtm_fixture(
    tmp_path: Path, *, vehicle: str = "MARKET"
) -> tuple[Path, dict[str, object]]:
    repository_root = Path(__file__).resolve().parents[1]
    corpus = tmp_path / "corpus"
    session = tmp_path / "session"
    module = tmp_path / "one_trade_bot.py"
    owner = "mtm:trainer-adversarial"
    _write_mtm_corpus(corpus)
    action = {
        "MARKET": (
            "            trade_id = self.broker.market_order(pair, 'LONG', 100)\n"
            "            self.broker.set_exit(trade_id, tp_price=151.0, "
            "sl_price=149.0)\n"
        ),
        "STOP_LOSS": (
            "            trade_id = self.broker.market_order(pair, 'LONG', 100)\n"
            "            self.broker.set_exit(trade_id, tp_price=151.0, "
            "sl_price=150.0)\n"
        ),
        "LIMIT": (
            "            self.broker.limit_order(pair, 'LONG', 100, "
            "price=150.0, tp_pips=25, sl_pips=25)\n"
        ),
        "STOP": (
            "            self.broker.stop_order(pair, 'LONG', 100, "
            "price=150.04, tp_pips=25, sl_pips=25)\n"
        ),
        "CANCEL": (
            "            order_id = self.broker.limit_order(pair, 'LONG', 100, "
            "price=140.0, tp_pips=25, sl_pips=25)\n"
            "            self.broker.cancel_order(order_id)\n"
        ),
    }[vehicle]
    module.write_text(
        "from quant_rabbit.dojo_lab_provenance import OwnedBrokerView\n"
        "class Bot:\n"
        "    def __init__(self, broker):\n"
        f"        self.broker = OwnedBrokerView(broker, {owner!r}, "
        "max_concurrent_per_pair=1, global_max_concurrent=2)\n"
        "        self.entered = False\n"
        "    def on_bar_closed(self, pair, bar, epoch):\n"
        "        if pair == 'USD_JPY' and not self.entered:\n"
        + action
        + "            self.entered = True\n",
        encoding="utf-8",
    )
    config_json = "{}"
    command = [
        sys.executable,
        str(repository_root / "scripts" / "run-virtual-market-session.py"),
        "--feed",
        "replay",
        "--session-dir",
        str(session),
        "--pairs",
        "CAD_JPY,USD_JPY",
        "--corpus-root",
        str(corpus),
        "--from",
        "2026-01-01T00:00:00Z",
        "--to",
        "2026-01-01T00:02:00Z",
        "--intrabar",
        "OHLC",
        "--bars-per-second",
        "10000",
        "--fast-ledger",
        "--continuous-mtm",
        "--bot-module",
        f"{module}:Bot",
        "--strategy-owner-id",
        owner,
        "--bot-dependency",
        "src/quant_rabbit/dojo_lab_provenance.py",
        "--bot-dependency",
        "src/quant_rabbit/virtual_broker.py",
        "--settle-at-end",
    ]
    environment = dict(os.environ)
    environment["DOJO_BOT_CONFIG"] = config_json
    completed = subprocess.run(
        command,
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    ledger = session / "ledger.jsonl"
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    manifest = records[0]["payload"]["reproducibility_manifest"]
    arguments: dict[str, object] = {
        "start_balance_jpy": 200_000.0,
        "expected_pairs": ["CAD_JPY", "USD_JPY"],
        "window_start": "2026-01-01T00:00:00Z",
        "window_end": "2026-01-01T00:02:00Z",
        "expected_intrabar": "OHLC",
        "expected_slippage_pips_per_fill": 0.0,
        "expected_financing_pips_per_day": 0.0,
        "expected_corpus_sha256": manifest["corpus"]["corpus_sha256"],
        "expected_bot_config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "expected_strategy_owner_id": owner,
        "expected_bot_module_sha256": manifest["bot"]["module_sha256"],
        "expected_bot_dependency_sha256": manifest["bot"]["dependency_sha256"],
        "expected_feed_pairs": ["CAD_JPY", "USD_JPY"],
        "expected_max_concurrent_per_pair": 1,
        "expected_global_max_concurrent": 2,
    }
    return ledger, arguments


def _score_strict_mtm(ledger: Path, arguments: dict[str, object]) -> dict:
    return score_ledger_metrics(ledger, **arguments)  # type: ignore[arg-type]


def _mtm_rows(ledger: Path) -> list[dict]:
    return [json.loads(line) for line in ledger.read_text().splitlines()]


def _reseal_mtm_marks(rows: list[dict]) -> None:
    previous_mark_sha = "0" * 64
    terminal_mark: dict | None = None
    for row in rows:
        if row["event"] != "ACCOUNT_MARK":
            continue
        mark = row["payload"]
        for state_name in ("account", "positions", "orders", "quotes"):
            mark[f"{state_name}_sha256"] = _canonical_sha(mark[state_name])
        mark["previous_mark_sha256"] = previous_mark_sha
        mark_body = {key: value for key, value in mark.items() if key != "mark_sha256"}
        mark["mark_sha256"] = _canonical_sha(mark_body)
        previous_mark_sha = mark["mark_sha256"]
        if mark["kind"] == "TERMINAL":
            terminal_mark = mark
    assert terminal_mark is not None
    stop = rows[-1]["payload"]
    stop["mtm_terminal_mark_sha256"] = terminal_mark["mark_sha256"]
    stop["account"] = json.loads(json.dumps(terminal_mark["account"]))


def test_coordinate_complete_mtm_contract_is_verified(tmp_path: Path) -> None:
    ledger, arguments = _strict_mtm_fixture(tmp_path)

    result = _score_strict_mtm(ledger, arguments)

    assert result["mtm_complete"] is True
    assert result["mtm_evidence_status"] == (
        "VERIFIED_COORDINATE_COMPLETE_ACCOUNT_MARK_CHAIN"
    )
    assert result["mtm_mark_count"] == 10
    assert result["mtm_max_drawdown_fraction"] is not None


@pytest.mark.parametrize(
    ("vehicle", "required_events"),
    [
        ("MARKET", {"FILL_MARKET", "SET_EXIT", "CLOSE"}),
        ("LIMIT", {"ORDER_LIMIT", "FILL_LIMIT", "CLOSE"}),
        ("STOP", {"ORDER_STOP", "FILL_LIMIT", "CLOSE"}),
        ("CANCEL", {"ORDER_LIMIT", "ORDER_CANCEL"}),
    ],
)
def test_mtm_verifier_reconstructs_order_and_exit_action_paths(
    tmp_path: Path, vehicle: str, required_events: set[str]
) -> None:
    ledger, arguments = _strict_mtm_fixture(tmp_path, vehicle=vehicle)
    observed_events = {row["event"] for row in _mtm_rows(ledger)}

    result = _score_strict_mtm(ledger, arguments)

    assert required_events.issubset(observed_events)
    assert result["mtm_complete"] is True
    assert result["terminal_flat"] is True


def test_mtm_contract_rejects_missing_phase_mark(tmp_path: Path) -> None:
    ledger, arguments = _strict_mtm_fixture(tmp_path)
    rows = _mtm_rows(ledger)
    phase_index = next(
        index
        for index, row in enumerate(rows)
        if row["event"] == "ACCOUNT_MARK" and row["payload"]["kind"] == "PHASE"
    )
    del rows[phase_index]
    _rewrite_ledger_chain(ledger, rows)

    with pytest.raises(DojoBotTrainerError, match="MTM_CONTRACT_VIOLATION"):
        _score_strict_mtm(ledger, arguments)


def test_mtm_contract_rejects_partial_pair_batch(tmp_path: Path) -> None:
    ledger, arguments = _strict_mtm_fixture(tmp_path)
    rows = _mtm_rows(ledger)
    batch = next(row for row in rows if row["event"] == "QUOTE_BATCH_BEGIN")
    batch["payload"]["batch_pairs"] = ["CAD_JPY"]
    batch["payload"]["coverage_complete"] = False
    _rewrite_ledger_chain(ledger, rows)

    with pytest.raises(DojoBotTrainerError, match="MTM_CONTRACT_VIOLATION"):
        _score_strict_mtm(ledger, arguments)


def test_mtm_contract_rejects_quote_coordinate_mismatch(tmp_path: Path) -> None:
    ledger, arguments = _strict_mtm_fixture(tmp_path)
    rows = _mtm_rows(ledger)
    batch = next(row for row in rows if row["event"] == "QUOTE_BATCH_BEGIN")
    batch["payload"]["quotes"][0]["ts"] = "2026-01-01T00:01:00+00:00#O"
    _rewrite_ledger_chain(ledger, rows)

    with pytest.raises(DojoBotTrainerError, match="MTM_CONTRACT_VIOLATION"):
        _score_strict_mtm(ledger, arguments)


def test_mtm_contract_rejects_bot_actions_moved_to_high_phase(tmp_path: Path) -> None:
    ledger, arguments = _strict_mtm_fixture(tmp_path, vehicle="CANCEL")
    rows = _mtm_rows(ledger)
    moved = []
    for owned_event in ("ORDER_LIMIT", "ORDER_CANCEL"):
        index = next(
            index for index, row in enumerate(rows) if row["event"] == owned_event
        )
        moved.append(rows.pop(index))
    high_batch_index = next(
        index
        for index, row in enumerate(rows)
        if row["event"] == "QUOTE_BATCH_BEGIN"
        and row["payload"]["coordinate"]["phase"] == "H"
        and row["payload"]["coordinate"]["epoch"] == 1_767_225_660
    )
    rows[high_batch_index + 1 : high_batch_index + 1] = moved
    _rewrite_ledger_chain(ledger, rows)

    with pytest.raises(DojoBotTrainerError, match="outside a causal O callback"):
        _score_strict_mtm(ledger, arguments)


def test_mtm_contract_rejects_new_order_filled_in_same_quote_batch(
    tmp_path: Path,
) -> None:
    ledger, arguments = _strict_mtm_fixture(tmp_path, vehicle="LIMIT")
    rows = _mtm_rows(ledger)
    fill_index = next(
        index for index, row in enumerate(rows) if row["event"] == "FILL_LIMIT"
    )
    fill = rows.pop(fill_index)
    order_index = next(
        index for index, row in enumerate(rows) if row["event"] == "ORDER_LIMIT"
    )
    rows.insert(order_index + 1, fill)
    _rewrite_ledger_chain(ledger, rows)

    with pytest.raises(
        DojoBotTrainerError, match="asynchronous broker event follows an O action"
    ):
        _score_strict_mtm(ledger, arguments)


def test_mtm_contract_rejects_omitted_touched_resting_fill_after_reseal(
    tmp_path: Path,
) -> None:
    ledger, arguments = _strict_mtm_fixture(tmp_path, vehicle="LIMIT")
    rows = _mtm_rows(ledger)
    fill_index = next(
        index for index, row in enumerate(rows) if row["event"] == "FILL_LIMIT"
    )
    fill = rows.pop(fill_index)
    next_mark = next(
        row
        for row in rows[fill_index:]
        if row["event"] == "ACCOUNT_MARK" and row["payload"]["kind"] == "PHASE"
    )["payload"]
    order_event = next(row for row in rows if row["event"] == "ORDER_LIMIT")["payload"]
    next_mark["positions"] = []
    next_mark["orders"] = [
        {
            "order_id": order_event["order_id"],
            "pair": order_event["pair"],
            "side": order_event["side"],
            "units": order_event["units"],
            "limit_price": order_event["price"],
            "tp_pips": order_event["tp_pips"],
            "sl_pips": order_event["sl_pips"],
            "kind": "LIMIT",
        }
    ]
    next_mark["account"] = {
        "balance_jpy": 200_000.0,
        "equity_jpy": 200_000.0,
        "margin_used_jpy": 0.0,
        "margin_usage": 0.0,
        "accrued_financing_jpy": 0.0,
        "open_positions": 0,
        "resting_orders": 1,
    }
    assert fill["payload"]["order_id"] == order_event["order_id"]
    _reseal_mtm_marks(rows)
    _rewrite_ledger_chain(ledger, rows)

    with pytest.raises(DojoBotTrainerError, match="omits mandatory broker consequence"):
        _score_strict_mtm(ledger, arguments)


def test_mtm_contract_rejects_omitted_touched_losing_stop_after_reseal(
    tmp_path: Path,
) -> None:
    ledger, arguments = _strict_mtm_fixture(tmp_path, vehicle="STOP_LOSS")
    rows = _mtm_rows(ledger)
    exit_index = next(
        index for index, row in enumerate(rows) if row["event"] == "EXIT_SL"
    )
    exit_row = rows.pop(exit_index)
    prior_mark = next(
        row["payload"]
        for row in reversed(rows[:exit_index])
        if row["event"] == "ACCOUNT_MARK" and row["payload"]["kind"] == "PHASE"
    )
    next_mark = next(
        row["payload"]
        for row in rows[exit_index:]
        if row["event"] == "ACCOUNT_MARK" and row["payload"]["kind"] == "PHASE"
    )
    position = json.loads(json.dumps(prior_mark["positions"][0]))
    quote = next(item for item in next_mark["quotes"] if item["pair"] == "USD_JPY")
    entry = float(position["entry_price"])
    units = float(position["units"])
    equity = 200_000.0 + (float(quote["bid"]) - entry) * units
    margin = units * ((float(quote["bid"]) + float(quote["ask"])) / 2.0) / 25.0
    next_mark["positions"] = [position]
    next_mark["account"] = {
        "balance_jpy": 200_000.0,
        "equity_jpy": round(equity, 2),
        "margin_used_jpy": round(margin, 2),
        "margin_usage": round(margin / equity, 6),
        "accrued_financing_jpy": 0.0,
        "open_positions": 1,
        "resting_orders": 0,
    }
    assert float(exit_row["payload"]["pl_jpy"]) < 0
    _reseal_mtm_marks(rows)
    _rewrite_ledger_chain(ledger, rows)

    with pytest.raises(DojoBotTrainerError, match="omits mandatory broker consequence"):
        _score_strict_mtm(ledger, arguments)


def test_mtm_contract_rejects_quote_not_present_in_sealed_corpus(
    tmp_path: Path,
) -> None:
    ledger, arguments = _strict_mtm_fixture(tmp_path)
    rows = _mtm_rows(ledger)
    batch = next(row for row in rows if row["event"] == "QUOTE_BATCH_BEGIN")
    batch["payload"]["quotes"][0]["bid"] += 0.001
    _rewrite_ledger_chain(ledger, rows)

    with pytest.raises(DojoBotTrainerError, match="sealed corpus bytes"):
        _score_strict_mtm(ledger, arguments)


def test_mtm_contract_rejects_missing_sealed_corpus_shard(tmp_path: Path) -> None:
    ledger, arguments = _strict_mtm_fixture(tmp_path)
    rows = _mtm_rows(ledger)
    manifest = rows[0]["payload"]["reproducibility_manifest"]
    root = Path(manifest["corpus"]["root"])
    shard = root / manifest["corpus"]["shards"][0]["path"]
    shard.unlink()

    with pytest.raises(DojoBotTrainerError, match="corpus shard is unavailable"):
        _score_strict_mtm(ledger, arguments)


def test_mtm_contract_rejects_self_consistent_false_account_mark(
    tmp_path: Path,
) -> None:
    ledger, arguments = _strict_mtm_fixture(tmp_path)
    rows = _mtm_rows(ledger)
    mark = next(
        row
        for row in rows
        if row["event"] == "ACCOUNT_MARK" and row["payload"]["kind"] == "TERMINAL"
    )
    mark["payload"]["account"]["balance_jpy"] += 1_000.0
    mark["payload"]["account"]["equity_jpy"] += 1_000.0
    mark["payload"]["account_sha256"] = _canonical_sha(mark["payload"]["account"])
    mark_body = {
        key: value for key, value in mark["payload"].items() if key != "mark_sha256"
    }
    mark["payload"]["mark_sha256"] = _canonical_sha(mark_body)
    rows[-1]["payload"]["account"] = json.loads(json.dumps(mark["payload"]["account"]))
    rows[-1]["payload"]["mtm_terminal_mark_sha256"] = mark["payload"]["mark_sha256"]
    _rewrite_ledger_chain(ledger, rows)

    with pytest.raises(
        DojoBotTrainerError, match=r"account\.balance_jpy is not reconstructed"
    ):
        _score_strict_mtm(ledger, arguments)


def test_mtm_contract_rejects_fake_terminal_claim(tmp_path: Path) -> None:
    ledger, arguments = _strict_mtm_fixture(tmp_path)
    rows = _mtm_rows(ledger)
    rows[-1]["payload"]["mtm_terminal_mark_sha256"] = "f" * 64
    _rewrite_ledger_chain(ledger, rows)

    with pytest.raises(DojoBotTrainerError, match="MTM_CONTRACT_VIOLATION"):
        _score_strict_mtm(ledger, arguments)


@pytest.mark.parametrize(
    ("vehicle", "owned_event"),
    [
        ("MARKET", "FILL_MARKET"),
        ("MARKET", "SET_EXIT"),
        ("LIMIT", "ORDER_LIMIT"),
        ("STOP", "ORDER_STOP"),
        ("CANCEL", "ORDER_CANCEL"),
    ],
)
def test_mtm_contract_rejects_owned_action_owner_rewrite(
    tmp_path: Path, vehicle: str, owned_event: str
) -> None:
    ledger, arguments = _strict_mtm_fixture(tmp_path, vehicle=vehicle)
    rows = _mtm_rows(ledger)
    action = next(row for row in rows if row["event"] == owned_event)
    action["payload"]["strategy_owner_id"] = "mtm:wrong-owner"
    _rewrite_ledger_chain(ledger, rows)

    with pytest.raises(DojoBotTrainerError, match="strategy owner"):
        _score_strict_mtm(ledger, arguments)
