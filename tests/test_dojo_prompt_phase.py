from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_rabbit.dojo_ai_discretion import (
    build_capability_manifest,
    build_day_packet,
    prelock_prompt,
    prelock_scorer,
    score_sealed_response,
    seal_answer_key,
    seal_model_manifest,
    seal_response,
)
from quant_rabbit.dojo_prompt_phase import (
    DojoPromptPhaseError,
    LOCKED_PREREGISTRATION_SHA256,
    LOCKED_VARIANT_PROMPT_SHA256,
    PHASE_RANKS,
    assert_locked_preregistration,
    build_cell_assignment,
    build_phase_manifest,
    score_prompt_phase,
)


REPO = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO / "research/registries/dojo_prompt_experiment_v1.json"
NOW = datetime(2026, 7, 19, tzinfo=timezone.utc)


def _registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _dummy_assignments() -> list[dict]:
    assignments = []
    for rank in PHASE_RANKS["phase_1_diagnostic"]:
        blind_day_id = _digest(f"blind-day-{rank}")
        source_sha = _digest(f"source-{rank}")
        for variant in LOCKED_VARIANT_PROMPT_SHA256:
            assignments.append(
                build_cell_assignment(
                    phase_id="phase_1_diagnostic",
                    blind_day_rank=rank,
                    blind_day_id=blind_day_id,
                    variant_id=variant,
                    source_sha256=source_sha,
                    packet_sha256=_digest(f"packet-{rank}-{variant}"),
                    prompt_sha256=LOCKED_VARIANT_PROMPT_SHA256[variant],
                    prompt_lock_sha256=_digest(f"prompt-lock-{rank}-{variant}"),
                    model_sha256=_digest(f"model-{rank}-{variant}"),
                    capability_manifest_sha256=_digest(f"capability-{rank}-{variant}"),
                    request_receipt_sha256=_digest(f"request-{rank}-{variant}"),
                    context_id=f"fresh-context-{rank}-{variant}",
                )
            )
    return assignments


def _real_score_and_assignment() -> tuple[dict, dict]:
    variant = "A_FABLE_MINIMAL"
    prompt_text = (REPO / "research/prompts/dojo_fable_minimal_v1.txt").read_text()
    capability = build_capability_manifest(
        context_id="real-cell-context", generated_at_utc=NOW
    )
    prompt = prelock_prompt(prompt_text, variant_id=variant, locked_at_utc=NOW)
    scorer = prelock_scorer(locked_at_utc=NOW)
    model = seal_model_manifest(
        model_name="diagnostic-model",
        model_version="v1",
        model_lineage="one-unattested-lineage",
        reasoning_effort="high",
        context_id="real-cell-context",
        capability_manifest=capability,
        locked_at_utc=NOW,
    )
    source = {
        "contract": "QR_DOJO_AI_DAY_SOURCE_V1",
        "blind_nonce": _digest("blind-nonce-real-cell"),
        "pair": "USD_JPY",
        "decision_cutoff_utc": "2026-01-02T00:00:00Z",
        "observations": [
            {
                "observed_at_utc": "2026-01-01T23:59:00Z",
                "kind": "QUOTE",
                "payload": {"bid": 150.0, "ask": 150.02},
            }
        ],
    }
    packet = build_day_packet(
        source,
        prompt_lock=prompt,
        capability_manifest=capability,
        scorer_lock=scorer,
    )
    response = seal_response(
        {
            "trial_id": packet["trial_id"],
            "action": "LONG",
            "pair": "USD_JPY",
            "size": "HALF",
            "confidence": 0.6,
            "evidence_refs": ["obs-001"],
            "target_pips": 10.0,
            "invalidation_pips": 8.0,
            "strongest_counterargument": "Range continuation could fail.",
            "abstain_reason": None,
        },
        packet=packet,
        prompt_lock=prompt,
        model_manifest=model,
        capability_manifest=capability,
        sealed_at_utc=NOW + timedelta(seconds=1),
    )
    answer = seal_answer_key(
        trial_id=packet["trial_id"],
        packet_sha256=packet["packet_sha256"],
        returns={
            "FLAT": 0.0,
            "LONG_HALF": 0.01,
            "LONG_FULL": 0.02,
            "SHORT_HALF": -0.01,
            "SHORT_FULL": -0.02,
        },
        sealed_at_utc=NOW + timedelta(seconds=2),
    )
    score = score_sealed_response(
        response,
        packet=packet,
        prompt_lock=prompt,
        model_manifest=model,
        capability_manifest=capability,
        scorer_lock=scorer,
        answer_key_loader=lambda: answer,
        opened_at_utc=NOW + timedelta(seconds=3),
    )
    assignment = build_cell_assignment(
        phase_id="phase_1_diagnostic",
        blind_day_rank=1,
        blind_day_id=_digest("blind-day-1"),
        variant_id=variant,
        source_sha256=packet["source_sha256"],
        packet_sha256=packet["packet_sha256"],
        prompt_sha256=prompt["prompt_sha256"],
        prompt_lock_sha256=prompt["prompt_lock_sha256"],
        model_sha256=model["model_sha256"],
        capability_manifest_sha256=capability["capability_manifest_sha256"],
        request_receipt_sha256=_digest("real-request"),
        context_id="real-cell-context",
    )
    return score, assignment


def test_locked_registry_and_runtime_prompt_hashes_are_byte_identical() -> None:
    registry = assert_locked_preregistration(_registry())
    assert (
        hashlib.sha256(
            json.dumps(
                registry,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        == LOCKED_PREREGISTRATION_SHA256
    )
    for variant in registry["variants"]:
        prompt = (REPO / variant["prompt_path"]).read_text(encoding="utf-8")
        lock = prelock_prompt(
            prompt, variant_id=variant["variant_id"], locked_at_utc=NOW
        )
        assert lock["prompt_sha256"] == variant["prompt_sha256"]


def test_manifest_requires_exact_90_cells_shared_day_and_fresh_contexts() -> None:
    assignments = _dummy_assignments()
    with pytest.raises(DojoPromptPhaseError, match="exactly 90"):
        build_phase_manifest(
            _registry(),
            phase_id="phase_1_diagnostic",
            assignments=assignments[:-1],
            locked_at_utc=NOW,
        )

    drifted = copy.deepcopy(assignments)
    drifted[1]["source_sha256"] = _digest("different-source")
    drifted[1] = build_cell_assignment(
        **{key: drifted[1][key] for key in drifted[1] if key != "cell_id"}
    )
    with pytest.raises(DojoPromptPhaseError, match="share one source"):
        build_phase_manifest(
            _registry(),
            phase_id="phase_1_diagnostic",
            assignments=drifted,
            locked_at_utc=NOW,
        )

    reused = copy.deepcopy(assignments)
    reused[1]["context_id"] = reused[0]["context_id"]
    with pytest.raises(DojoPromptPhaseError, match="context is reused"):
        build_phase_manifest(
            _registry(),
            phase_id="phase_1_diagnostic",
            assignments=reused,
            locked_at_utc=NOW,
        )


def test_all_missing_cells_remain_90_failures_and_never_become_genuine_flat() -> None:
    manifest = build_phase_manifest(
        _registry(),
        phase_id="phase_1_diagnostic",
        assignments=_dummy_assignments(),
        locked_at_utc=NOW,
    )
    result = score_prompt_phase(
        _registry(),
        manifest,
        scored_cells={},
        terminal_failures={},
        sealed_at_utc=NOW + timedelta(days=1),
    )

    assert result["allocated_cell_count"] == 90
    assert result["response_failure_cell_count"] == 90
    assert result["valid_response_cell_count"] == 0
    assert result["missing_response_is_genuine_flat"] is False
    assert result["prompt_selection_allowed"] is False
    for summary in result["variant_summaries"].values():
        assert summary["allocated_cell_count"] == 30
        assert summary["response_failure_cell_count"] == 30
        assert summary["genuine_flat_count"] == 0
        assert summary["synthetic_flat_failure_count"] == 30
        assert summary["compounded_net_return"] == 0.0


def test_variants_are_scored_separately_and_paired_within_day() -> None:
    score, real_assignment = _real_score_and_assignment()
    assignments = _dummy_assignments()
    real_source = real_assignment["source_sha256"]
    for index, assignment in enumerate(assignments):
        if assignment["blind_day_rank"] == 1:
            if assignment["variant_id"] == "A_FABLE_MINIMAL":
                assignments[index] = real_assignment
            else:
                assignment = dict(assignment)
                assignment["source_sha256"] = real_source
                assignments[index] = build_cell_assignment(
                    **{key: assignment[key] for key in assignment if key != "cell_id"}
                )
    manifest = build_phase_manifest(
        _registry(),
        phase_id="phase_1_diagnostic",
        assignments=assignments,
        locked_at_utc=NOW,
    )
    result = score_prompt_phase(
        _registry(),
        manifest,
        scored_cells={real_assignment["cell_id"]: score},
        terminal_failures={},
        sealed_at_utc=NOW + timedelta(days=1),
    )

    a = result["variant_summaries"]["A_FABLE_MINIMAL"]
    b = result["variant_summaries"]["B_CALIBRATED_ABSTENTION"]
    c = result["variant_summaries"]["C_STRUCTURAL_REGIME"]
    assert a["total_log_growth"] == pytest.approx(math.log1p(0.01))
    assert a["compounded_net_return"] == pytest.approx(0.01)
    assert b["compounded_net_return"] == 0.0
    assert c["compounded_net_return"] == 0.0
    day_one = result["paired_day_results"][0]
    assert day_one["b_minus_a_log_growth"] == pytest.approx(-math.log1p(0.01))
    assert day_one["c_minus_a_log_growth"] == pytest.approx(-math.log1p(0.01))
    assert day_one["all_three_valid"] is False
    assert result["paired_contrasts"]["confirmatory_inference_allowed"] is False
    assert result["positive_superiority_claim_allowed"] is False


def test_registry_drift_and_score_parent_drift_fail_closed() -> None:
    registry = _registry()
    registry["score_policy"]["missing_response_is_failure"] = False
    with pytest.raises(DojoPromptPhaseError, match="preregistration bytes drifted"):
        assert_locked_preregistration(registry)

    score, real_assignment = _real_score_and_assignment()
    assignments = _dummy_assignments()
    for index, assignment in enumerate(assignments):
        if assignment["blind_day_rank"] == 1:
            if assignment["variant_id"] == "A_FABLE_MINIMAL":
                assignments[index] = real_assignment
            else:
                assignment = dict(assignment)
                assignment["source_sha256"] = real_assignment["source_sha256"]
                assignments[index] = build_cell_assignment(
                    **{key: assignment[key] for key in assignment if key != "cell_id"}
                )
    manifest = build_phase_manifest(
        _registry(),
        phase_id="phase_1_diagnostic",
        assignments=assignments,
        locked_at_utc=NOW,
    )
    wrong_cell = next(
        cell
        for cell in manifest["cells"]
        if cell["variant_id"] == "A_FABLE_MINIMAL" and cell["blind_day_rank"] == 2
    )
    with pytest.raises(DojoPromptPhaseError, match="packet_sha256"):
        score_prompt_phase(
            _registry(),
            manifest,
            scored_cells={wrong_cell["cell_id"]: score},
            terminal_failures={},
            sealed_at_utc=NOW + timedelta(days=1),
        )


def test_phase_cli_locks_and_scores_all_90_terminal_cells(tmp_path: Path) -> None:
    assignment_paths = []
    for index, assignment in enumerate(_dummy_assignments()):
        path = tmp_path / f"assignment-{index:03d}.json"
        path.write_text(json.dumps(assignment), encoding="utf-8")
        assignment_paths.append(path)
    manifest_path = tmp_path / "manifest.json"
    score_path = tmp_path / "score.json"
    script = REPO / "scripts/run-dojo-ai-experiment.py"
    environment = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    manifest_command = [
        sys.executable,
        str(script),
        "build-manifest",
        "--phase-id",
        "phase_1_diagnostic",
        "--locked-at-utc",
        NOW.isoformat(),
        "--output",
        str(manifest_path),
    ]
    for path in assignment_paths:
        manifest_command.extend(("--assignment", str(path)))
    built = subprocess.run(
        manifest_command,
        cwd=REPO,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    built_result = json.loads(built.stdout)
    assert built_result["allocated_cell_count"] == 90

    scored = subprocess.run(
        [
            sys.executable,
            str(script),
            "score",
            "--manifest",
            str(manifest_path),
            "--sealed-at-utc",
            (NOW + timedelta(days=1)).isoformat(),
            "--output",
            str(score_path),
        ],
        cwd=REPO,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    scored_result = json.loads(scored.stdout)
    artifact = json.loads(score_path.read_text(encoding="utf-8"))
    assert scored_result["response_failure_cell_count"] == 90
    assert artifact["terminal_cell_count"] == 90
    assert artifact["prompt_selection_allowed"] is False
    assert score_path.stat().st_mode & 0o777 == 0o600
