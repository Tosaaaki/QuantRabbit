from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import build_paper_research_jpy_oracle_checkpoint_v1 as checkpoint


ROOT = Path(__file__).resolve().parent


def copy_inputs(destination: Path) -> None:
    for relative in (
        checkpoint.ORACLE_PATH,
        checkpoint.VERIFIER_PATH,
        checkpoint.CONTRACT_PATH,
        checkpoint.SCHEMA_PATH,
        checkpoint.VERIFIER_SCHEMA_PATH,
        checkpoint.ORACLE_TEST_PATH,
        checkpoint.VERIFIER_TEST_PATH,
        checkpoint.CHECKPOINT_TEST_PATH,
        checkpoint.BUILDER_PATH,
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for cycle in checkpoint.SEALED_CYCLES:
        relative = Path(f"evidence/orchestrator_state_v2/official_seal_v{cycle}.json")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)


def assert_boundary(payload: dict) -> None:
    assert payload["checkpoint_id"] == "PAPER_RESEARCH_JPY_ORACLE_V1"
    assert payload["classification"] == "FUTURE_ONLY_INDEPENDENT_ECONOMIC_ORACLE"
    assert payload["producer_metrics_used"] is False
    assert payload["same_signal_ids_all_arms"] is True
    assert payload["all_proposals_have_all_arm_dispositions"] is True
    assert payload["terminal_inventory_mtm_jpy_micros"] == 0
    assert payload["external_orders"] == 0
    assert payload["holdout_state"] == "UNOPENED"
    assert payload["official_strategy_run_performed"] is False
    assert payload["profit_evidence_generated"] is False
    assert payload["anchor_status"] == "LOCAL_REPRODUCIBLE"
    assert payload["remote_anchor_required_for_external_status"] is True
    assert payload["legacy_official_oracle_pass_count"] == 0
    assert payload["legacy_seals_changed"] is False
    assert payload["authority"]["paper_only"] is True
    assert payload["authority"]["live_authority"] is False
    assert payload["authority"]["credential_access"] is False
    assert payload["audit_sha256"] == checkpoint.embedded(payload, "audit_sha256")


def test_builder_is_deterministic_and_tamper_evident(tmp_path: Path) -> None:
    root = tmp_path / "copy"
    copy_inputs(root)
    first = checkpoint.build(root)
    assert_boundary(first)
    assert checkpoint.validate(root) == first
    first_text = (root / checkpoint.AUDIT_PATH).read_text()
    assert checkpoint.build(root) == first
    assert (root / checkpoint.AUDIT_PATH).read_text() == first_text

    ledger = root / checkpoint.LEDGER_PATH
    ledger.write_text(ledger.read_text() + "\n")
    with pytest.raises(checkpoint.EvidenceError, match="evidence changed"):
        checkpoint.validate(root)


def test_legacy_coverage_is_sidecar_only_and_hash_binds_seals(tmp_path: Path) -> None:
    root = tmp_path / "copy"
    copy_inputs(root)
    checkpoint.build(root)
    coverage = json.loads((root / checkpoint.LEGACY_COVERAGE_PATH).read_text())
    assert coverage["sealed_cycle_count"] == 13
    assert coverage["reconstructable_count"] == 0
    assert coverage["official_oracle_pass_count"] == 0
    assert coverage["legacy_seals_changed"] is False
    for row in coverage["cycles"]:
        assert row["official_oracle_pass"] is False
        assert row["retroactive_promotion_allowed"] is False
        assert row["coverage_state"] in {"RETROACTIVE", "MISSING"}
        assert row["missing_independent_oracle_inputs"]
        if row["legacy_seal_path"]:
            assert checkpoint.sha256_file(root / row["legacy_seal_path"]) == row[
                "legacy_seal_sha256"
            ]


def test_checked_in_oracle_checkpoint_reproduces() -> None:
    payload = checkpoint.validate(ROOT)
    assert_boundary(payload)
    for relative, expected in payload["evidence_artifact_sha256"].items():
        assert checkpoint.sha256_file(ROOT / relative) == expected
