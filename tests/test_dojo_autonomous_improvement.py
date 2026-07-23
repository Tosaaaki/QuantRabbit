from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_rabbit.dojo_autonomous_improvement import (
    CANDIDATE_SPEC_CONTRACT,
    SHADOW_ASSESSMENT_CONTRACT,
    SHADOW_OUTCOME_CONTRACT,
    SHADOW_OUTCOME_CONTRACT_V2,
    DojoAutonomousEvidenceError,
    append_candidate_event,
    append_shadow_assessment,
    append_shadow_outcome,
    build_candidate_spec,
    build_shadow_assessment,
    build_shadow_outcome,
    initialize_research_root,
    validate_research_root,
)


NOW = datetime(2026, 7, 22, 18, 0, tzinfo=timezone.utc)
SHA = "a" * 64


def _guard() -> dict:
    return {
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }


def _assessment() -> dict:
    return {
        "contract": SHADOW_ASSESSMENT_CONTRACT,
        **_guard(),
        "as_of_utc": NOW.isoformat(),
        "horizon_end_utc": (NOW + timedelta(hours=1)).isoformat(),
        "ledger_sha256": SHA,
        "state_sha256": "b" * 64,
        "snapshot_sha256": "c" * 64,
        "pair": "EUR_USD",
        "strategy_family": "W_FADE",
        "regime": "TREND",
        "facts": ["24h change positive", "6h efficiency rising"],
        "primary_path": "trend continuation",
        "alternative_path": "range re-entry",
        "falsifier": "M5 closes below bound support",
        "habitat_reason": "fade conflicts with efficient trend",
        "confidence": 0.7,
        "supervision": "CAUTION",
        "quote": {
            "pair": "EUR_USD",
            "bid": 1.14,
            "ask": 1.14008,
            "ts_utc": (NOW - timedelta(seconds=1)).isoformat(),
        },
        "source_watermarks": [
            {
                "kind": "m1_bid_ask",
                "observed_through_utc": (
                    NOW - timedelta(seconds=1)
                ).isoformat(),
                "sha256": "d" * 64,
            }
        ],
        "positions": [
            {
                "position_id": "T1",
                "entry_context_sha256": "e" * 64,
                "opened_at_utc": (NOW - timedelta(minutes=30)).isoformat(),
                "units": 1000,
                "entry_price": 1.139,
                "executable_mark": 1.14,
                "unrealized_pnl_jpy": 100.0,
                "tp_progress": 0.4,
                "ceiling_remaining_minutes": 450,
                "margin_usage": 0.1,
                "capital_lock_jpy": 8000.0,
                "thesis": "WOUNDED",
                "inventory": "CONCENTRATED",
                "shadow_action": "NO_NEW_ENTRY_TEST",
            }
        ],
    }


def _outcome(assessment_id: str) -> dict:
    return {
        "contract": SHADOW_OUTCOME_CONTRACT,
        **_guard(),
        "assessment_id": assessment_id,
        "observed_through_utc": (NOW + timedelta(hours=1)).isoformat(),
        "settled_at_utc": None,
        "realized_pnl_jpy": -25.0,
        "mfe_pips": 2.0,
        "mae_pips": -5.0,
        "actual_exit_price": 1.1395,
        "counterfactual_exit_price": 1.14,
        "counterfactual_delta_jpy": 20.0,
        "regime_correct": True,
    }


def _multi_assessment() -> dict:
    assessment = _assessment()
    assessment["positions"].append(
        {
            "position_id": "T2",
            "entry_context_sha256": "f" * 64,
            "opened_at_utc": (NOW - timedelta(minutes=20)).isoformat(),
            "units": 800,
            "entry_price": 1.141,
            "executable_mark": 1.14008,
            "unrealized_pnl_jpy": 80.0,
            "tp_progress": 0.3,
            "ceiling_remaining_minutes": 460,
            "margin_usage": 0.15,
            "capital_lock_jpy": 6000.0,
            "thesis": "ALIVE",
            "inventory": "TRAPPED",
            "shadow_action": "OBSERVE_HOLD",
        }
    )
    return assessment


def _multi_outcome(assessment_id: str) -> dict:
    observed = (NOW + timedelta(hours=1)).isoformat()
    position_outcomes = [
        {
            "position_id": "T1",
            "side": "LONG",
            "status": "HORIZON_MARK",
            "observed_through_utc": observed,
            "settled_at_utc": None,
            "realized_pnl_jpy": 110.0,
            "mfe_pips": 4.0,
            "mae_pips": -2.0,
            "actual_exit_price": 1.1401,
            "counterfactual_exit_price": 1.14,
            "counterfactual_delta_jpy": 10.0,
        },
        {
            "position_id": "T2",
            "side": "SHORT",
            "status": "HORIZON_MARK",
            "observed_through_utc": observed,
            "settled_at_utc": None,
            "realized_pnl_jpy": 70.0,
            "mfe_pips": 3.0,
            "mae_pips": -1.0,
            "actual_exit_price": 1.1402,
            "counterfactual_exit_price": 1.14008,
            "counterfactual_delta_jpy": -10.0,
        },
    ]
    return {
        "contract": SHADOW_OUTCOME_CONTRACT_V2,
        **_guard(),
        "assessment_id": assessment_id,
        "observed_through_utc": observed,
        "portfolio_pnl_jpy": 180.0,
        "portfolio_counterfactual_delta_jpy": 0.0,
        "regime_correct": True,
        "position_outcomes": position_outcomes,
    }


def _spec() -> dict:
    return {
        "contract": CANDIDATE_SPEC_CONTRACT,
        **_guard(),
        "family": "INVENTORY_RELEASE",
        "hypothesis": "release only invalidated inventory",
        "causal_narrative": "large ceiling losses dominate small wins",
        "expected_mechanism": "free capital without censoring later winners",
        "falsifier": "independent stress expectancy is non-positive",
        "affected_pair": "EUR_USD",
        "affected_strategy": "W_FADE",
        "evidence_cohort": "tagged-settlements-through-bound-tip",
        "changed_rule": {
            "name": "invalidated_exit",
            "baseline": False,
            "candidate": True,
        },
        "unchanged_controls": ["entry", "size", "tp", "ceiling"],
        "evidence_sha256s": [SHA],
        "windows": {
            "TRAIN": {
                "from_utc": "2026-01-01T00:00:00+00:00",
                "to_utc": "2026-02-01T00:00:00+00:00",
                "source_sha256": "1" * 64,
            },
            "VAL": {
                "from_utc": "2026-02-01T00:00:00+00:00",
                "to_utc": "2026-03-01T00:00:00+00:00",
                "source_sha256": "2" * 64,
            },
            "S5": {
                "from_utc": "2026-03-01T00:00:00+00:00",
                "to_utc": "2026-04-01T00:00:00+00:00",
                "source_sha256": "3" * 64,
            },
        },
        "costs": {
            "BASE": {
                "slippage_pips_per_fill": 0.0,
                "financing_pips_per_day": 0.0,
            },
            "STRESS": {
                "slippage_pips_per_fill": 0.3,
                "financing_pips_per_day": 0.8,
            },
        },
        "intrabar_paths": ["OHLC", "OLHC"],
        "end_of_replay_forced_close_benefit": False,
        "risk_gates": {"min_independent_stress_pf": 1.25},
        "death_codes": [
            "COST",
            "DIRECTION",
            "EXIT_TIMING",
            "INVENTORY",
            "MEASUREMENT",
            "OVERFIT",
            "REGIME_MISMATCH",
            "RISK",
        ],
    }


def test_shadow_is_cutoff_bound_and_outcome_waits_for_maturity(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ai_shadow_ledger.jsonl"
    row, appended = append_shadow_assessment(
        ledger, _assessment(), recorded_at_utc=NOW
    )
    assert appended
    assessment = row["payload"]
    with pytest.raises(DojoAutonomousEvidenceError, match="not mature"):
        append_shadow_outcome(
            ledger,
            _outcome(assessment["assessment_id"]),
            recorded_at_utc=NOW + timedelta(minutes=59),
        )
    outcome_row, appended = append_shadow_outcome(
        ledger,
        _outcome(assessment["assessment_id"]),
        recorded_at_utc=NOW + timedelta(hours=1),
    )
    assert appended and outcome_row["event_type"] == "OUTCOME_RECORDED"
    with pytest.raises(DojoAutonomousEvidenceError, match="already exists"):
        append_shadow_outcome(
            ledger,
            _outcome(assessment["assessment_id"]),
            recorded_at_utc=NOW + timedelta(hours=2),
        )


def test_shadow_rejects_future_source_and_live_authority() -> None:
    future = _assessment()
    future["source_watermarks"][0]["observed_through_utc"] = (
        NOW + timedelta(seconds=1)
    ).isoformat()
    with pytest.raises(DojoAutonomousEvidenceError, match="after as_of"):
        build_shadow_assessment(future)
    live = _assessment()
    live["order_authority"] = "WRITE"
    with pytest.raises(DojoAutonomousEvidenceError, match="authority NONE"):
        build_shadow_assessment(live)


def test_shadow_rejects_hindsight_backfill(tmp_path: Path) -> None:
    with pytest.raises(DojoAutonomousEvidenceError, match="hindsight"):
        append_shadow_assessment(
            tmp_path / "shadow.jsonl",
            _assessment(),
            recorded_at_utc=NOW + timedelta(minutes=6),
        )


def test_multi_position_shadow_requires_v2_and_scores_each_position(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ai_shadow_ledger.jsonl"
    assessment_row, appended = append_shadow_assessment(
        ledger, _multi_assessment(), recorded_at_utc=NOW
    )
    assert appended
    assessment_id = assessment_row["payload"]["assessment_id"]
    with pytest.raises(
        DojoAutonomousEvidenceError, match="requires outcome V2"
    ):
        append_shadow_outcome(
            ledger,
            _outcome(assessment_id),
            recorded_at_utc=NOW + timedelta(hours=1),
        )
    outcome_row, appended = append_shadow_outcome(
        ledger,
        _multi_outcome(assessment_id),
        recorded_at_utc=NOW + timedelta(hours=1),
    )
    assert appended
    outcome = outcome_row["payload"]
    assert outcome["contract"] == SHADOW_OUTCOME_CONTRACT_V2
    assert [item["position_id"] for item in outcome["position_outcomes"]] == [
        "T1",
        "T2",
    ]
    assert validate_research_root(tmp_path)["shadow"]["status"] == "VALID"


def test_multi_position_shadow_rejects_missing_identity_and_bad_totals() -> None:
    assessment = build_shadow_assessment(_multi_assessment())
    missing = _multi_outcome(assessment["assessment_id"])
    missing["position_outcomes"].pop()
    with pytest.raises(
        DojoAutonomousEvidenceError, match="exactly cover"
    ):
        build_shadow_outcome(
            missing,
            assessment=assessment,
            recorded_at_utc=NOW + timedelta(hours=1),
        )
    bad_total = _multi_outcome(assessment["assessment_id"])
    bad_total["portfolio_pnl_jpy"] = 181.0
    with pytest.raises(
        DojoAutonomousEvidenceError, match="totals do not match"
    ):
        build_shadow_outcome(
            bad_total,
            assessment=assessment,
            recorded_at_utc=NOW + timedelta(hours=1),
        )


def test_multi_position_shadow_rejects_early_or_future_scoring() -> None:
    assessment = build_shadow_assessment(_multi_assessment())
    payload = _multi_outcome(assessment["assessment_id"])
    with pytest.raises(DojoAutonomousEvidenceError, match="not mature"):
        build_shadow_outcome(
            payload,
            assessment=assessment,
            recorded_at_utc=NOW + timedelta(minutes=59),
        )
    future = _multi_outcome(assessment["assessment_id"])
    future["position_outcomes"][0]["observed_through_utc"] = (
        NOW + timedelta(hours=2)
    ).isoformat()
    with pytest.raises(
        DojoAutonomousEvidenceError, match="outside the scoring cutoff"
    ):
        build_shadow_outcome(
            future,
            assessment=assessment,
            recorded_at_utc=NOW + timedelta(hours=1),
        )


def test_candidate_spec_is_one_change_with_separate_windows() -> None:
    sealed = build_candidate_spec(_spec())
    assert len(sealed["candidate_id"]) == 64
    broken = _spec()
    broken["changed_rule"]["second_change"] = 2
    with pytest.raises(DojoAutonomousEvidenceError, match="exactly one change"):
        build_candidate_spec(broken)
    overlap = _spec()
    overlap["windows"]["VAL"]["from_utc"] = "2026-01-15T00:00:00+00:00"
    with pytest.raises(DojoAutonomousEvidenceError, match="overlap"):
        build_candidate_spec(overlap)


def test_candidate_lifecycle_blocks_competitor_and_weak_pass(
    tmp_path: Path,
) -> None:
    initialize_research_root(
        tmp_path, recorded_at_utc=NOW, implementation_sha256=SHA
    )
    event = {**_guard(), "spec": _spec()}
    row, appended = append_candidate_event(
        tmp_path,
        event_type="CANDIDATE_PREREGISTERED",
        payload=event,
        recorded_at_utc=NOW + timedelta(seconds=1),
    )
    assert appended
    candidate_id = row["payload"]["candidate_id"]
    _, bootstrap_appended = initialize_research_root(
        tmp_path,
        recorded_at_utc=NOW + timedelta(seconds=2),
        implementation_sha256=SHA,
    )
    assert not bootstrap_appended
    assert validate_research_root(tmp_path)["active_candidate"] == {
        "candidate_id": candidate_id,
        "status": "PREREGISTERED",
    }
    same_row, same_appended = append_candidate_event(
        tmp_path,
        event_type="CANDIDATE_PREREGISTERED",
        payload=event,
        recorded_at_utc=NOW + timedelta(seconds=2),
    )
    assert not same_appended and same_row == row
    competitor = _spec()
    competitor["changed_rule"]["candidate"] = "different"
    with pytest.raises(DojoAutonomousEvidenceError, match="active candidate"):
        append_candidate_event(
            tmp_path,
            event_type="CANDIDATE_PREREGISTERED",
            payload={**_guard(), "spec": competitor},
            recorded_at_utc=NOW + timedelta(seconds=2),
        )
    started = {
        **_guard(),
        "candidate_id": candidate_id,
        "job_lock": {
            "git_head_sha256": "4" * 64,
            "spec_sha256": row["payload"]["spec"]["spec_sha256"],
            "policy_sha256": "5" * 64,
            "output_manifest_sha256": "6" * 64,
            "argv": ["python3", "replay.py"],
            "environment_allowlist": ["PYTHONPATH"],
            "output_directory": "/tmp/dojo-candidate-output",
            "screen_name": "qr-dojo-improve-abcd1234",
            "pid": 1234,
            "process_command_sha256": "7" * 64,
        },
    }
    append_candidate_event(
        tmp_path,
        event_type="REPLAY_STARTED",
        payload=started,
        recorded_at_utc=NOW + timedelta(seconds=3),
    )
    failed = {
        **_guard(),
        "candidate_id": candidate_id,
        "failure_code": "MEASUREMENT",
        "reason": "replay metrics used append wall clock",
        "artifact_sha256": "8" * 64,
    }
    append_candidate_event(
        tmp_path,
        event_type="REPLAY_FAILED",
        payload=failed,
        recorded_at_utc=NOW + timedelta(seconds=4),
    )
    assert validate_research_root(tmp_path)["active_candidate"] == {
        "candidate_id": candidate_id,
        "status": "FAILED",
    }
    retry = {
        **started,
        "job_lock": {
            **started["job_lock"],
            "git_head_sha256": "9" * 64,
            "output_manifest_sha256": "a" * 64,
            "process_command_sha256": "b" * 64,
            "pid": 5678,
        },
    }
    append_candidate_event(
        tmp_path,
        event_type="REPLAY_RETRY_STARTED",
        payload=retry,
        recorded_at_utc=NOW + timedelta(seconds=5),
    )
    assert validate_research_root(tmp_path)["active_candidate"] == {
        "candidate_id": candidate_id,
        "status": "STARTED",
    }
    weak = {
        **_guard(),
        "candidate_id": candidate_id,
        "independent_stress_metrics": {
            "pf": 1.1,
            "net": 10.0,
            "expectancy": 1.0,
            "worst_day_not_worse": True,
            "drawdown_not_worse": True,
            "margin_ruin_not_worse": True,
            "unresolved_end_exposure": False,
        },
    }
    with pytest.raises(DojoAutonomousEvidenceError, match="did not pass"):
        append_candidate_event(
            tmp_path,
            event_type="REPLAY_PASSED",
            payload=weak,
            recorded_at_utc=NOW + timedelta(seconds=6),
        )
    rejected = {
        **_guard(),
        "candidate_id": candidate_id,
        "death_code": "RISK",
        "reason": "stress PF below gate",
    }
    append_candidate_event(
        tmp_path,
        event_type="REPLAY_REJECTED",
        payload=rejected,
        recorded_at_utc=NOW + timedelta(seconds=7),
    )
    assert validate_research_root(tmp_path)["active_candidate"] is None


def test_tampered_candidate_chain_fails_closed(tmp_path: Path) -> None:
    initialize_research_root(
        tmp_path, recorded_at_utc=NOW, implementation_sha256=SHA
    )
    ledger = tmp_path / "candidate_ledger.jsonl"
    row = json.loads(ledger.read_text().strip())
    row["payload"]["order_authority"] = "WRITE"
    ledger.write_text(json.dumps(row) + "\n")
    with pytest.raises(DojoAutonomousEvidenceError, match="chain invalid"):
        validate_research_root(tmp_path)
