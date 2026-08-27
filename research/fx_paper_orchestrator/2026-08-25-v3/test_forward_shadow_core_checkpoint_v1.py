from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import build_forward_shadow_core_checkpoint_v1 as checkpoint
import forward_shadow_core_v1 as shadow


ROOT = Path(__file__).resolve().parent


def copy_builder_inputs(destination: Path) -> None:
    for relative in (
        checkpoint.POLICY_PATH,
        checkpoint.RUNTIME_PATH,
        checkpoint.BUILDER_PATH,
        checkpoint.CORE_TEST_PATH,
        checkpoint.CHECKPOINT_TEST_PATH,
        "jpy_accounting_v2.py",
        *checkpoint.PROTECTED_PATHS,
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)


def assert_checkpoint_boundary(payload: dict) -> None:
    assert payload["checkpoint_id"] == "CREDENTIAL_FREE_FORWARD_SHADOW_CORE_V1"
    assert payload["classification"] == (
        "NON_STRATEGY_FILE_ONLY_FORWARD_SHADOW_INFRASTRUCTURE"
    )
    assert payload["authority"] == shadow.AUTHORITY
    assert payload["forward_feed_connected"] is False
    assert payload["forward_observation_started"] is False
    assert payload["official_strategy_run_performed"] is False
    assert payload["profit_evidence_generated"] is False
    assert payload["strategy_adoption_authorized"] is False
    assert payload["holdout_state"] == "UNOPENED"
    assert payload["external_orders"] == 0
    assert payload["input_scope"]["network_transport_present"] is False
    assert payload["input_scope"]["secret_source_present"] is False
    assert payload["input_scope"]["external_endpoint_present"] is False
    assert payload["input_scope"]["source_unchanged_after_ingest"] is True
    assert payload["batch_manifest"]["exact_duplicate_count"] == 1
    assert payload["batch_manifest"]["lossless"] is True
    assert payload["causal_completed_bars"]["counts"] == {
        "M5": 100,
        "M15": 32,
        "H1": 8,
        "H4": 2,
    }
    assert payload["quality_failure_matrix"]["sequence_absent_lossless_false"] is True
    assert payload["quality_failure_matrix"]["price_imputation_used"] is False
    assert {
        "SOURCE_OR_ARRIVAL_GAP",
        "HEARTBEAT_FAILURE",
        "RECONNECT_BOUNDARY",
        "OUT_OF_ORDER_EVENT",
        "CLOCK_REVERSAL",
        "SPREAD_INVERSION",
    } <= set(payload["quality_failure_matrix"]["covered_halts"])
    execution = payload["shared_proposal_execution"]
    assert execution["actual_llm_called"] is False
    assert execution["same_content_addressed_proposal_all_arms"] is True
    assert execution["external_order_count"] == 0
    final = payload["paper_account_finalization"]
    assert final["terminal_inventory_mtm_jpy"] == 0.0
    assert final["terminal_currency_inventory"] == {}
    assert final["max_age_close_count"] == 4
    assert final["terminal_liquidation_count"] == 4
    assert payload["restart_safety"]["state_and_checkpoint_hash_match"] is True
    assert payload["restart_safety"]["exact_batch_reingest_idempotent"] is True
    assert payload["audit_sha256"] == checkpoint.embedded_hash(
        payload, "audit_sha256"
    )


def test_build_validate_roundtrip_is_deterministic_and_tamper_evident(
    tmp_path: Path,
) -> None:
    root = tmp_path / "shadow_copy"
    copy_builder_inputs(root)
    built = checkpoint.build(root)
    assert_checkpoint_boundary(built)
    validated = checkpoint.validate(root)
    assert validated == built
    first_text = (root / checkpoint.AUDIT_PATH).read_text(encoding="utf-8")
    assert checkpoint.build(root) == built
    assert (root / checkpoint.AUDIT_PATH).read_text(encoding="utf-8") == first_text

    manifest_path = root / checkpoint.VALID_MANIFEST_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["event_count"] += 1
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(checkpoint.ShadowEvidenceError, match="evidence changed"):
        checkpoint.validate(root)


def test_source_hash_change_invalidates_frozen_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "shadow_copy"
    copy_builder_inputs(root)
    checkpoint.build(root)
    runtime = root / checkpoint.RUNTIME_PATH
    runtime.write_text(runtime.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(checkpoint.ShadowEvidenceError, match="evidence changed"):
        checkpoint.validate(root)


def test_checked_in_shadow_checkpoint_is_reproducible_and_boundary_closed() -> None:
    payload = checkpoint.validate(ROOT)
    assert_checkpoint_boundary(payload)
    for relative, expected in payload["artifact_file_sha256"].items():
        assert checkpoint.sha256_file(ROOT / relative) == expected
