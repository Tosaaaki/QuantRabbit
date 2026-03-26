from __future__ import annotations

import json
from pathlib import Path

from scripts import preflight_guard


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_validate_improvement_artifact_accepts_allow_new_lane(tmp_path: Path) -> None:
    artifact = _write_json(
        tmp_path / "improvement_gate_latest.json",
        {
            "query": "repeat-risk check",
            "blocked": False,
            "candidates": [
                {
                    "candidate": {"strategy": "MomentumBurst"},
                    "action": "allow_new_lane",
                    "reasons": ["no unresolved overlap found for this candidate"],
                }
            ],
        },
    )

    ok, details, payload = preflight_guard._validate_improvement_artifact(
        artifact,
        expected_query="repeat-risk check",
        max_age_sec=3600,
        now_ts=artifact.stat().st_mtime + 10,
    )

    assert ok is True
    assert payload["blocked"] is False
    assert "improvement artifact query: repeat-risk check" in details


def test_validate_improvement_artifact_rejects_blocked_payload(tmp_path: Path) -> None:
    artifact = _write_json(
        tmp_path / "improvement_gate_latest.json",
        {
            "query": "repeat-risk check",
            "blocked": True,
            "recommended_single_focus_lane": {
                "hypothesis_key": "microlevelreactor_bounce_lower_expected_pips_contra_surface_20260316",
                "primary_loss_driver": "negative_forecast_long_scaled_not_rejected",
            },
            "candidates": [
                {
                    "candidate": {"strategy": "MicroLevelReactor"},
                    "action": "review_existing_pending",
                    "reasons": [
                        "same strategy already has unresolved trading lane",
                        "validate recommended single-focus lane before opening another same-strategy tweak",
                    ],
                }
            ],
        },
    )

    ok, details, _ = preflight_guard._validate_improvement_artifact(
        artifact,
        expected_query="repeat-risk check",
        max_age_sec=3600,
        now_ts=artifact.stat().st_mtime + 10,
    )

    assert ok is False
    assert any(
        "latest improvement_preflight is blocked" in detail for detail in details
    )
    assert any("recommended_single_focus_lane:" in detail for detail in details)
    assert any("blocked_candidate: MicroLevelReactor" in detail for detail in details)


def test_validate_improvement_artifact_rejects_query_mismatch(tmp_path: Path) -> None:
    artifact = _write_json(
        tmp_path / "improvement_gate_latest.json",
        {
            "query": "old query",
            "blocked": False,
            "candidates": [
                {
                    "candidate": {"strategy": "MomentumBurst"},
                    "action": "allow_new_lane",
                    "reasons": ["no unresolved overlap found for this candidate"],
                }
            ],
        },
    )

    ok, details, _ = preflight_guard._validate_improvement_artifact(
        artifact,
        expected_query="new query",
        max_age_sec=3600,
        now_ts=artifact.stat().st_mtime + 10,
    )

    assert ok is False
    assert any("improvement artifact query mismatch" in detail for detail in details)
