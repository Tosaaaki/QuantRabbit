#!/usr/bin/env python3
"""Build deterministic local evidence for the independent JPY Oracle V2.

This builder does not import the Oracle, verifier, or launcher. It writes a
small ex-ante accounting fixture, executes both pinned runtimes through the
fixed sealed-FD launcher, and copies the exact receipts into a content-addressed
checkpoint. The result is accounting-only plumbing evidence, never strategy,
profitability, or holdout-admission evidence.
"""

from __future__ import annotations

import argparse
import ast
import base64
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import secrets
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

ORACLE_PATH = "paper_research_jpy_oracle_v2.py"
VERIFIER_PATH = "paper_research_oracle_verifier_v2.py"
LAUNCHER_PATH = "paper_research_fd_launcher_v2.py"
GOLDEN_PATH = "paper_research_jpy_oracle_golden_v2.py"
REFERENCE_PATH = "paper_research_double_entry_reference_v2.py"
REFERENCE_CONTRACT_PATH = "PAPER_RESEARCH_DOUBLE_ENTRY_REFERENCE_CONTRACT_V2.json"
REFERENCE_TEST_PATH = "test_paper_research_double_entry_reference_v2.py"
REFERENCE_MUTATION_TEST_PATH = (
    "test_paper_research_double_entry_reference_v2_runtime_mutations.py"
)
CONTRACT_PATH = "PAPER_RESEARCH_JPY_ORACLE_CONTRACT_V2.json"
SCHEMA_PATH = "paper_research_jpy_oracle_schema_v2.json"
VERIFIER_SCHEMA_PATH = "paper_research_oracle_verifier_schema_v2.json"
ORACLE_TEST_PATH = "test_paper_research_jpy_oracle_v2.py"
ORACLE_FINANCE_TEST_PATH = (
    "test_paper_research_jpy_oracle_v2_finance_adversarial.py"
)
VERIFIER_TEST_PATH = "test_paper_research_oracle_verifier_v2.py"
LAUNCHER_TEST_PATH = "test_paper_research_fd_launcher_v2.py"
CHECKPOINT_TEST_PATH = "test_paper_research_jpy_oracle_corrective_v2.py"
BUILDER_PATH = "build_paper_research_jpy_oracle_corrective_v2.py"
BUILDER_LOCK_PATH = "paper_research_jpy_oracle_corrective_v2.lock"

SOURCE_FILES = (
    CONTRACT_PATH,
    SCHEMA_PATH,
    ORACLE_PATH,
    VERIFIER_SCHEMA_PATH,
    VERIFIER_PATH,
    GOLDEN_PATH,
    REFERENCE_PATH,
    REFERENCE_CONTRACT_PATH,
    REFERENCE_TEST_PATH,
    REFERENCE_MUTATION_TEST_PATH,
    LAUNCHER_PATH,
    ORACLE_TEST_PATH,
    ORACLE_FINANCE_TEST_PATH,
    VERIFIER_TEST_PATH,
    LAUNCHER_TEST_PATH,
    BUILDER_LOCK_PATH,
    BUILDER_PATH,
    CHECKPOINT_TEST_PATH,
)

EVIDENCE_ROOT = "evidence/paper_research_jpy_oracle_corrective_v2"
AUDIT_PATH = f"{EVIDENCE_ROOT}/oracle_checkpoint_v2.json"
LEGACY_COVERAGE_PATH = f"{EVIDENCE_ROOT}/legacy_oracle_coverage_v2.json"
SUPERSESSION_PATH = f"{EVIDENCE_ROOT}/superseded_oracle_v1.json"
ORACLE_LAUNCH_RECEIPT_PATH = f"{EVIDENCE_ROOT}/oracle_launcher_receipt_v2.json"
VERIFIER_LAUNCH_RECEIPT_PATH = f"{EVIDENCE_ROOT}/verifier_launcher_receipt_v2.json"
CHECKPOINT_COMMIT_PATH = f"{EVIDENCE_ROOT}/CHECKPOINT_COMMIT.json"

START_NS = 1_767_225_600_000_000_000
SEALED_CYCLES = (25, 27, 28, 29, 30, 31, 33, 35, 37, 38, 39, 40, 41)
FAILED_CYCLES = (26, 32, 36)
INVALID_CYCLES = (34,)
FAILED_CYCLE_RECEIPTS = {
    26: "V26_AUTHORIZED_RECOVERY_FAILURE.json",
    32: "V32_OFFICIAL_EXECUTION_FAILURE.json",
    36: "V36_OFFICIAL_EXECUTION_FAILURE.json",
}
LEGACY_RUN_ARTIFACTS = {
    25: (
        "evidence/run_asian_usd_coherence_persistence_v25_official_001/"
        "result_asian_usd_coherence_persistence_v25.json",
        "evidence/run_asian_usd_coherence_persistence_v25_official_001/"
        "proposal_ledger_asian_usd_coherence_persistence_v25.jsonl",
    ),
    27: (
        "evidence/run_causal_min_spread_representative_v27_official_001/"
        "result_causal_min_spread_representative_v27.json",
        "evidence/run_causal_min_spread_representative_v27_official_001/"
        "proposal_ledger_causal_min_spread_representative_v27.jsonl",
    ),
    28: (
        "evidence/run_causal_basket_hold_v28_official_001/"
        "result_causal_basket_hold_v28.json",
        "evidence/run_causal_basket_hold_v28_official_001/"
        "proposal_ledger_causal_basket_hold_v28.jsonl",
    ),
    29: (
        "evidence/run_causal_basket_consensus_release_v29_official_001/"
        "result_causal_basket_consensus_release_v29.json",
        "evidence/run_causal_basket_consensus_release_v29_official_001/"
        "proposal_ledger_causal_basket_consensus_release_v29.jsonl",
    ),
    30: (
        "evidence/run_causal_consensus_release_scope_v30_official_001/"
        "result_causal_consensus_release_scope_v30.json",
        "evidence/run_causal_consensus_release_scope_v30_official_001/"
        "proposal_ledger_causal_consensus_release_scope_v30.jsonl",
    ),
    31: (
        "evidence/run_causal_consensus_release_persistence_v31_official_001/"
        "result_causal_consensus_release_persistence_v31.json",
        "evidence/run_causal_consensus_release_persistence_v31_official_001/"
        "proposal_ledger_causal_consensus_release_persistence_v31.jsonl",
    ),
    33: (
        "evidence/run_asian_displacement_handoff_fade_v33_official_001/"
        "result_asian_displacement_handoff_fade_v33.json",
        "evidence/run_asian_displacement_handoff_fade_v33_official_001/"
        "proposal_ledger_asian_displacement_handoff_fade_v33.jsonl",
    ),
    34: (
        "evidence/run_causal_tail_excess_representative_v34_official_001/"
        "result_causal_tail_excess_representative_v34.json",
        "evidence/run_causal_tail_excess_representative_v34_official_001/"
        "proposal_ledger_causal_tail_excess_representative_v34.jsonl",
    ),
    35: (
        "evidence/run_global_no_overlap_admission_v35_official_001/"
        "result_global_no_overlap_admission_v35.json",
        "evidence/run_global_no_overlap_admission_v35_official_001/"
        "proposal_ledger_global_no_overlap_admission_v35.jsonl",
    ),
    37: (
        "evidence/run_london_asian_range_breakout_v37_official_001/"
        "result_london_asian_range_breakout_v37.json",
        "evidence/run_london_asian_range_breakout_v37_official_001/"
        "proposal_ledger_london_asian_range_breakout_v37.jsonl",
    ),
    38: (
        "evidence/run_london_overextension_fade_v38_official_001/"
        "result_london_overextension_fade_v38.json",
        "evidence/run_london_overextension_fade_v38_official_001/"
        "proposal_ledger_london_overextension_fade_v38.jsonl",
    ),
    39: (
        "evidence/run_london_overextension_carry_v39_official_001/"
        "result_london_overextension_carry_v39.json",
        "evidence/run_london_overextension_carry_v39_official_001/"
        "proposal_ledger_london_overextension_carry_v39.jsonl",
    ),
    40: (
        "evidence/run_london_fix_overextension_fade_v40_official_001/"
        "result_london_fix_overextension_fade_v40.json",
        "evidence/run_london_fix_overextension_fade_v40_official_001/"
        "proposal_ledger_london_fix_overextension_fade_v40.jsonl",
    ),
    41: (
        "evidence/run_london_open_false_break_reclaim_v41_official_001/"
        "result_london_open_false_break_reclaim_v41.json",
        "evidence/run_london_open_false_break_reclaim_v41_official_001/"
        "proposal_ledger_london_open_false_break_reclaim_v41.jsonl",
    ),
}
LEGACY_FROZEN_ARTIFACT_COUNT = 45
LEGACY_FROZEN_ARTIFACT_STREAM_SHA256 = (
    "7bf9e87f06d2fcf5113535998f8cce673fdb71e04ca998c24d8ed3ef1e9e09f1"
)
LEGACY_V1_EVIDENCE_NAMES = (
    "accounting_policy_fixture_v1.json",
    "evaluation_policy_fixture_v1.json",
    "ex_ante_proposal_fixture_v1.json",
    "execution_policy_fixture_v1.json",
    "inventory_policy_fixture_v1.json",
    "legacy_oracle_coverage_v1.json",
    "oracle_checkpoint_v1.json",
    "oracle_ledger_v1.jsonl",
    "oracle_manifest_v1.json",
    "oracle_verifier_receipt_v1.json",
    "source_bbo_fixture_v1.jsonl",
    "source_bbo_manifest_fixture_v1.json",
)
SUPERSEDED_COMMIT = "9a2fbde9d107a9be08f666ef1edb6decdd1fc78c"
SUPERSEDED_COMMIT_TREE = "53207511fe98c99e9732f31e42f2009a0db24d83"
SUPERSEDED_V1_SUBTREE = "0b3eb8baf691d5a242ffb138ccf35669318706fc"
SUPERSEDED_V1_AGGREGATE_SHA256 = (
    "50d244688efd9ae5ce646e54d0d574525ebdc828ee5febc416bb1298ce2ab9ef"
)
SUPERSEDED_V1_AUDIT_SHA256 = (
    "58f5099ba73291a71e0b173a9b91f5438fe7f317aa2823fbcc4078a9d3a0af52"
)
SUPERSEDED_V1_AUDIT_FILE_SHA256 = (
    "252fb271d8ca42a95d9743759a6b38bd5438c12a0877baf5292ba59c851f182c"
)
SUPERSEDED_V1_GIT_PATH = (
    "research/fx_paper_orchestrator/2026-08-25-v3/"
    "evidence/paper_research_jpy_oracle_v1"
)
LEGACY_V1_AUTHORITY = {
    "paper_only": True,
    "live_authority": False,
    "broker_account_access": False,
    "credential_access": False,
    "order_endpoint": False,
    "external_orders": 0,
    "deploy": False,
}
LEGACY_V1_AUDIT_KEYS = frozenset({
    "schema_version", "checkpoint_id", "classification",
    "source_artifact_sha256", "evidence_artifact_sha256", "oracle_root_sha256",
    "verifier_receipt_sha256", "producer_metrics_used",
    "same_signal_ids_all_arms", "all_proposals_have_all_arm_dispositions",
    "terminal_inventory_mtm_jpy_micros", "external_orders", "holdout_state",
    "official_strategy_run_performed", "profit_evidence_generated",
    "anchor_status", "remote_anchor_required_for_external_status",
    "legacy_coverage_sha256", "legacy_official_oracle_pass_count",
    "legacy_seals_changed", "authority", "audit_sha256",
})
GIT = Path("/usr/bin/git")
RUNTIME_CLASSIFICATION = (
    "FUTURE_ONLY_ACCOUNTING_ONLY_LOCAL_UNANCHORED_NOT_ADMISSIBLE"
)
RUNTIME_ANCHOR_STATUS = "LOCAL_UNANCHORED"
EXECUTION_PROVENANCE_SCOPE = (
    "LOCAL_CALLER_ASSERTED_CONTENT_BINDING_NOT_EXECUTION_ATTESTATION_"
    "NOT_EXTERNALLY_ANCHORED"
)
REFERENCE_ENGINE_ID = "EVENT_SOURCED_DOUBLE_ENTRY_REFERENCE_V1"
REFERENCE_JOURNAL_TRANSACTION_COUNT = 22

# These exact runtime output schemas are deliberately centralized.  Oracle or
# verifier schema evolution must update one reviewed surface instead of
# weakening the builder to an open-ended ``dict.get`` acceptance path.
ORACLE_OUTPUT_FILES = frozenset({
    "intent.json", "oracle_ledger.jsonl", "oracle_manifest.json", "COMMIT.json",
})
ORACLE_INTENT_KEYS = frozenset({
    "schema_version", "transaction_id", "request_sha256", "code_sha256",
    "contract_sha256", "schema_sha256",
})
ORACLE_COMMIT_KEYS = frozenset({
    "schema_version", "transaction_id", "request_sha256", "intent_sha256",
    "ledger_sha256", "ledger_size_bytes", "manifest_sha256",
    "manifest_size_bytes", "terminal_hash",
})
ORACLE_MANIFEST_KEYS = frozenset({
    "schema_version", "oracle_implementation", "status", "classification",
    "causal_signal_admission", "release_evidence_eligible",
    "detector_replay_receipt_required", "authority",
    "oracle_release_content_binding", "oracle_execution_provenance_scope",
    "request_sha256", "input_artifact_sha256", "raw_source_manifest_sha256",
    "proposal_provenance_root_sha256", "producer_result_or_metrics_used",
    "proposal_identity_generated_by_oracle", "oracle_ledger_file",
    "oracle_ledger_sha256", "oracle_ledger_size_bytes",
    "oracle_ledger_row_count", "oracle_ledger_terminal_hash", "oracle_metrics",
    "terminal_inventory_mtm_jpy_micros", "external_orders", "anchor_status",
    "oracle_root_sha256",
})
ORACLE_METRICS_KEYS = frozenset({
    "schema_version", "initial_equity_jpy_micros", "same_signal_ids_all_arms",
    "all_proposals_have_all_arm_dispositions", "common_gross_reference_shared",
    "arms", "external_orders", "terminal_inventory_mtm_jpy_micros",
    "metrics_sha256",
})
ORACLE_ARM_METRICS_KEYS = frozenset({
    "proposal_count", "executed_count", "disposition_counts",
    "signal_id_set_sha256", "common_gross_pnl_jpy_micros",
    "realized_cost_jpy_micros", "fill_sizing_drag_jpy_micros",
    "latency_spread_slippage_drag_jpy_micros",
    "direct_commission_financing_cost_jpy_micros",
    "admission_opportunity_drag_jpy_micros",
    "total_execution_and_admission_drag_jpy_micros",
    "net_pnl_jpy_micros", "ending_equity_jpy_micros",
    "ending_equity_multiple", "direction_accuracy",
    "max_drawdown_jpy_micros", "max_drawdown_ratio", "cvar_tail_bps",
    "cluster_cvar_jpy_micros", "cluster_cvar_return",
    "currency_time_cluster_n_eff", "currency_time_cluster_observations",
    "monthly", "max_gross_notional_jpy_micros",
    "minimum_marked_equity_jpy_micros",
    "maximum_required_margin_jpy_micros",
    "minimum_free_margin_jpy_micros", "margin_guard_pass",
    "terminal_open_positions", "terminal_inventory_mtm_jpy_micros",
})
ORACLE_CLUSTER_OBSERVATION_KEYS = frozenset({
    "cluster_id", "time_bucket", "currency_nodes",
    "source_signal_set_sha256", "ledger_net_pnl_jpy_micros",
    "cluster_risk_net_pnl_jpy_micros", "signed_return",
})
ORACLE_MONTHLY_METRICS_KEYS = frozenset({
    "month_id", "comparable_full_month", "segment_start_ts_ns",
    "segment_end_ts_ns", "start_equity_jpy_micros",
    "end_equity_jpy_micros", "equity_multiple",
    "equity_multiple_status", "ruin_observed",
})
ORACLE_RELEASE_CONTENT_BINDING_KEYS = frozenset({
    "code_sha256", "contract_sha256", "schema_sha256", "launcher_sha256",
    "snapshot_mode",
})
VERIFIER_RELEASE_CONTENT_BINDING_KEYS = frozenset({
    "code_sha256", "schema_sha256", "launcher_sha256",
    "reference_code_sha256", "reference_contract_sha256",
    "reference_result_sha256", "snapshot_mode",
})
VERIFIER_OUTPUT_FILES = frozenset({"verifier_receipt.json", "COMMIT.json"})
VERIFIER_COMMIT_KEYS = frozenset({
    "schema_version", "request_sha256", "receipt_sha256", "receipt_size_bytes",
    "verifier_receipt_sha256",
})
VERIFIER_RECEIPT_KEYS = frozenset({
    "schema_version", "verifier_implementation", "status", "classification",
    "causal_signal_admission", "release_evidence_eligible", "admission_eligible",
    "detector_replay_receipt_required", "authority", "oracle_root_sha256",
    "oracle_manifest_sha256", "oracle_manifest_size_bytes", "oracle_ledger_sha256",
    "oracle_ledger_size_bytes", "expected_canonical_ledger_sha256",
    "oracle_ledger_terminal_hash", "raw_source_manifest_sha256",
    "oracle_request_sha256", "oracle_release_content_binding",
    "oracle_execution_provenance_scope", "verifier_release_content_binding",
    "verifier_execution_provenance_scope", "input_artifact_sha256",
    "independently_rebuilt_ledger", "independently_rebuilt_metrics",
    "producer_result_or_metrics_used", "verified_oracle_metrics",
    "terminal_inventory_mtm_jpy_micros", "external_orders", "anchor_status",
    "reference_engine_id", "reference_code_sha256",
    "reference_contract_sha256", "reference_input_root_sha256",
    "reference_journal_root_sha256", "reference_journal_transaction_count",
    "reference_all_transactions_balanced",
    "reference_economic_projection_sha256",
    "reference_result_sha256",
    "reference_accounting_diagnostics_only",
    "reference_n_eff_statistical_admission_allowed",
    "reference_direction_accuracy_profit_gate_allowed",
    "verifier_receipt_sha256",
})
REFERENCE_RESULT_SNAPSHOT_KEYS = frozenset({
    "engine_id", "input_root_sha256", "ledger_bytes_base64",
    "ledger_row_count", "ledger_terminal_hash", "oracle_metrics",
    "proposal_provenance_root_sha256", "journal_root_sha256",
    "journal_transaction_count", "all_transactions_balanced",
    "economic_projection_sha256",
})
ORACLE_INPUT_LABELS = (
    "source_blob", "source_manifest", "proposal", "execution_policy",
    "inventory_policy", "accounting_policy", "evaluation_policy",
    "instrument_registry", "authority_policy",
)
VERIFIER_INPUT_LABELS = (
    *ORACLE_INPUT_LABELS,
    "oracle_request", "oracle_code_snapshot", "oracle_contract_snapshot",
    "oracle_schema_snapshot", "reference_code_snapshot",
    "reference_contract_snapshot", "oracle_intent", "oracle_commit",
    "oracle_ledger", "oracle_manifest",
)
CHECKPOINT_AUDIT_KEYS = frozenset({
    "schema_version", "checkpoint_id", "classification",
    "source_artifact_sha256", "evidence_artifact_sha256", "oracle_root_sha256",
    "verifier_receipt_sha256", "launcher_sha256", "launcher_runtime_provenance",
    "golden_reference", "sealed_fd_execution",
    "runtime_native_exclusive_publication", "checkpoint_publication",
    "checkpoint_terminal_commit_required", "checkpoint_commit_path",
    "release_evidence_eligible", "local_reproducible_only",
    "outer_launch_provenance_status", "runtime_environment_scope",
    "strategy_admission_eligible", "producer_metrics_used",
    "same_signal_ids_all_arms", "all_proposals_have_all_arm_dispositions",
    "terminal_inventory_mtm_jpy_micros", "external_orders", "holdout_state",
    "official_strategy_run_performed", "profit_evidence_generated",
    "anchor_status", "remote_anchor_verified",
    "external_review_required_before_commit",
    "pre_external_review_commit_push_allowed", "legacy_coverage_sha256",
    "supersession_sha256", "legacy_official_oracle_pass_count",
    "legacy_seals_changed", "superseded_checkpoint_commit",
    "superseded_checkpoint_classification", "authority", "audit_sha256",
})
CHECKPOINT_COMMIT_KEYS = frozenset({
    "schema_version", "checkpoint_id", "classification", "audit_sha256",
    "artifact_count", "artifact_sha256", "artifact_set_sha256",
    "publication_state", "strategy_admission_eligible", "external_orders",
    "checkpoint_commit_sha256",
})
PAPER_ONLY_AUTHORITY = {
    "paper_only": True,
    "live_authority": False,
    "broker_account_access": False,
    "credential_access": False,
    "order_endpoint": False,
    "external_orders": 0,
    "deploy": False,
    "external_config_mutation": False,
}
AUTHORITY_POLICY_KEYS = frozenset({
    "schema_version", "policy_id", *PAPER_ONLY_AUTHORITY,
    "authority_policy_sha256",
})
PYTHON = Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3")
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)

MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_CODE_BYTES = 64 * 1024 * 1024
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024

ORACLE_INPUT_FILES = frozenset({
    "inputs/source_blob.jsonl",
    "inputs/source_manifest.json",
    "inputs/proposal.json",
    "inputs/execution_policy.json",
    "inputs/inventory_policy.json",
    "inputs/accounting_policy.json",
    "inputs/evaluation_policy.json",
    "inputs/instrument_registry.json",
    "inputs/authority_policy.json",
    "inputs/oracle_request.json",
})
ORACLE_OUTPUT_ROOT_FILES = frozenset({
    ".oracle_output.lock",
    *(f"oracle_output/{name}" for name in ORACLE_OUTPUT_FILES),
})
VERIFIER_INPUT_FILES = frozenset({
    *ORACLE_INPUT_FILES,
    "inputs/oracle_code_snapshot.py",
    "inputs/oracle_contract_snapshot.json",
    "inputs/oracle_schema_snapshot.json",
    "inputs/reference_code_snapshot.py",
    "inputs/reference_contract_snapshot.json",
    "inputs/verifier_request.json",
    *(f"oracle_output/{name}" for name in ORACLE_OUTPUT_FILES),
})
VERIFIER_OUTPUT_ROOT_FILES = frozenset({
    ".verifier_output.lock",
    *(f"verifier_output/{name}" for name in VERIFIER_OUTPUT_FILES),
})
RUNTIME_ARTIFACT_FILES = frozenset({
    *ORACLE_INPUT_FILES,
    "inputs/oracle_code_snapshot.py",
    "inputs/oracle_contract_snapshot.json",
    "inputs/oracle_schema_snapshot.json",
    "inputs/reference_code_snapshot.py",
    "inputs/reference_contract_snapshot.json",
    "inputs/verifier_request.json",
    *(f"oracle_output/{name}" for name in ORACLE_OUTPUT_FILES),
    *(f"verifier_output/{name}" for name in VERIFIER_OUTPUT_FILES),
})
PRE_AUDIT_ARTIFACT_FILES = frozenset({
    *(f"{EVIDENCE_ROOT}/{relative}" for relative in RUNTIME_ARTIFACT_FILES),
    ORACLE_LAUNCH_RECEIPT_PATH,
    VERIFIER_LAUNCH_RECEIPT_PATH,
    LEGACY_COVERAGE_PATH,
    SUPERSESSION_PATH,
})
NONTERMINAL_ARTIFACT_FILES = frozenset({*PRE_AUDIT_ARTIFACT_FILES, AUDIT_PATH})
TOTAL_ARTIFACT_FILES = frozenset({
    *NONTERMINAL_ARTIFACT_FILES,
    CHECKPOINT_COMMIT_PATH,
})
EXPECTED_RUNTIME_ARTIFACT_COUNT = 22
EXPECTED_PRE_AUDIT_ARTIFACT_COUNT = 26
EXPECTED_NONTERMINAL_ARTIFACT_COUNT = 27
EXPECTED_TOTAL_ARTIFACT_COUNT = 28


class EvidenceError(RuntimeError):
    """Fail-closed checkpoint construction or validation error."""


# Darwin <sys/acl.h> defines ACL_TYPE_EXTENDED as 0x00000100.
ACL_TYPE_EXTENDED = 0x00000100


def _bind_extended_acl_api(
    platform_name: str | None = None,
    library: Any | None = None,
) -> tuple[Any, Any, Any]:
    platform_name = sys.platform if platform_name is None else platform_name
    if platform_name != "darwin":
        raise EvidenceError("extended ACL inspection is unavailable on this host")
    try:
        if library is None:
            library = ctypes.CDLL(
                "/usr/lib/libSystem.B.dylib",
                use_errno=True,
            )
        get_acl = library.acl_get_fd_np
        free_acl = library.acl_free
    except (OSError, AttributeError) as error:
        raise EvidenceError("extended ACL inspection API is unavailable") from error
    get_acl.argtypes = (ctypes.c_int, ctypes.c_int)
    get_acl.restype = ctypes.c_void_p
    free_acl.argtypes = (ctypes.c_void_p,)
    free_acl.restype = ctypes.c_int
    return library, get_acl, free_acl


_ACL_LIBRARY, _ACL_GET_FD_NP, _ACL_FREE = _bind_extended_acl_api()


def _require_no_extended_acl_fd(descriptor: int, label: str) -> None:
    ctypes.set_errno(0)
    acl_pointer = _ACL_GET_FD_NP(descriptor, ACL_TYPE_EXTENDED)
    saved_errno = ctypes.get_errno()
    if acl_pointer:
        ctypes.set_errno(0)
        free_result = _ACL_FREE(acl_pointer)
        free_errno = ctypes.get_errno()
        if free_result != 0:
            raise EvidenceError(
                f"{label} ACL release failed with errno {free_errno}"
            )
        raise EvidenceError(f"{label} has an extended ACL")
    if saved_errno != errno.ENOENT:
        raise EvidenceError(
            f"{label} ACL inspection failed with errno {saved_errno}"
        )


def _directory_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_mode, info.st_uid


def _regular_identity(
    info: os.stat_result,
) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
        info.st_mode,
        info.st_uid,
    )


def _require_safe_directory(info: os.stat_result, label: str) -> None:
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() \
            or info.st_mode & 0o022:
        raise EvidenceError(f"{label} is not an owner-controlled directory")


def _require_safe_regular(
    info: os.stat_result,
    label: str,
    *,
    allowed_nlinks: frozenset[int] = frozenset({1}),
) -> None:
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() \
            or info.st_nlink not in allowed_nlinks or info.st_mode & 0o022:
        raise EvidenceError(f"{label} is not an owner-controlled regular file")


def _safe_directory_fd(descriptor: int, label: str) -> os.stat_result:
    info = os.fstat(descriptor)
    _require_safe_directory(info, label)
    _require_no_extended_acl_fd(descriptor, label)
    return info


def _safe_regular_fd(
    descriptor: int,
    label: str,
    *,
    allowed_nlinks: frozenset[int] = frozenset({1}),
) -> os.stat_result:
    info = os.fstat(descriptor)
    _require_safe_regular(info, label, allowed_nlinks=allowed_nlinks)
    _require_no_extended_acl_fd(descriptor, label)
    return info


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_regular_bytes(
    path: Path,
    label: str = "source artifact",
    *,
    maximum_bytes: int = MAX_ARTIFACT_BYTES,
) -> bytes:
    if type(maximum_bytes) is not int or maximum_bytes < 0:
        raise EvidenceError(f"{label} read bound is invalid")
    try:
        before = os.lstat(path)
    except OSError as error:
        raise EvidenceError(f"{label} cannot be inspected: {path}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise EvidenceError(f"{label} is not a regular non-symlink file: {path}")
    _require_safe_regular(before, f"{label}: {path}")
    if before.st_size < 0 or before.st_size > maximum_bytes:
        raise EvidenceError(f"{label} exceeds its byte bound: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = _safe_regular_fd(descriptor, f"opened {label}: {path}")
        if _regular_identity(before) != _regular_identity(opened):
            raise EvidenceError(f"{label} changed while opening: {path}")
        chunks: list[bytes] = []
        offset = 0
        while offset < opened.st_size:
            chunk = os.pread(descriptor, min(1024 * 1024, opened.st_size - offset), offset)
            if not chunk:
                raise EvidenceError(f"{label} was truncated while reading: {path}")
            chunks.append(chunk)
            offset += len(chunk)
        if os.pread(descriptor, 1, opened.st_size):
            raise EvidenceError(f"{label} grew while reading: {path}")
        after = _safe_regular_fd(descriptor, f"reread {label}: {path}")
        if _regular_identity(after) != _regular_identity(opened):
            raise EvidenceError(f"{label} changed while reading: {path}")
        try:
            named = os.lstat(path)
        except OSError as error:
            raise EvidenceError(f"{label} pathname changed while reading: {path}") from error
        _require_safe_regular(named, f"pathname {label}: {path}")
        if _regular_identity(named) != _regular_identity(opened):
            raise EvidenceError(f"{label} pathname changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def sha256_file(path: Path) -> str:
    return sha256_bytes(
        read_regular_bytes(path, maximum_bytes=MAX_CODE_BYTES)
    )


def _load_golden_payload(source_root: Path, expected_sha256: str) -> dict[str, Any]:
    """Execute the exact reviewed golden bytes without relying on import paths."""
    source_bytes = _read_root_relative_regular_bytes(
        source_root,
        GOLDEN_PATH,
        "golden reference",
        maximum_bytes=MAX_CODE_BYTES,
    )
    if sha256_bytes(source_bytes) != expected_sha256:
        raise EvidenceError("golden reference changed after source freeze")
    namespace: dict[str, Any] = {
        "__name__": "_paper_research_jpy_oracle_golden_v2_frozen",
        "__file__": GOLDEN_PATH,
        "__package__": None,
    }
    try:
        compiled = compile(source_bytes, GOLDEN_PATH, "exec", dont_inherit=True)
        exec(compiled, namespace, namespace)
        builder = namespace["build_golden_payload"]
        payload = builder()
    except Exception as error:
        raise EvidenceError("frozen golden reference could not be evaluated") from error
    if type(payload) is not dict or set(payload) != {"inputs", "expected"}:
        raise EvidenceError("frozen golden reference returned an invalid payload")
    return payload


def embedded(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return sha256_bytes(canonical(unsigned))


def seal(payload: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = embedded(result, field)
    return result


def _strict_json_object(value: bytes, label: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise EvidenceError(f"{label} contains a duplicate JSON key")
            result[key] = item
        return result

    def reject_constant(token: str) -> None:
        raise EvidenceError(f"{label} contains a non-finite JSON number")

    try:
        parsed = json.loads(
            value,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise EvidenceError(f"{label} is not valid JSON") from error
    if type(parsed) is not dict:
        raise EvidenceError(f"{label} must be a JSON object")
    return parsed


def _canonical_json_object(
    value: bytes,
    label: str,
    expected_keys: frozenset[str] | set[str] | None = None,
) -> dict[str, Any]:
    payload = _strict_json_object(value, label)
    if value != canonical(payload) + b"\n":
        raise EvidenceError(f"{label} is not exact canonical newline JSON")
    if expected_keys is not None and set(payload) != set(expected_keys):
        raise EvidenceError(f"{label} schema changed")
    return payload


def _exact_int(value: Any, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        raise EvidenceError(f"{label} must be exact integer {expected}")


def _exact_bool(value: Any, expected: bool, label: str) -> None:
    if type(value) is not bool or value is not expected:
        raise EvidenceError(f"{label} must be exact boolean {expected}")


def _fixed_decimal_text(value: Any, label: str) -> str:
    if type(value) is not str:
        raise EvidenceError(f"{label} must be a fixed-point decimal string")
    unsigned = value[1:] if value.startswith("-") else value
    whole, separator, fraction = unsigned.partition(".")
    if not separator or not whole.isdigit() or len(fraction) != 18 \
            or not fraction.isdigit():
        raise EvidenceError(f"{label} must have exactly 18 decimal places")
    if (len(whole) > 1 and whole.startswith("0")) or (
        value.startswith("-0") and set(unsigned.replace(".", "")) == {"0"}
    ):
        raise EvidenceError(f"{label} is not canonical fixed-point text")
    return value


def _validate_exact_authority(value: Any, label: str) -> None:
    if type(value) is not dict or set(value) != set(PAPER_ONLY_AUTHORITY):
        raise EvidenceError(f"{label} authority schema mismatch")
    for field, expected in PAPER_ONLY_AUTHORITY.items():
        if field == "external_orders":
            _exact_int(value[field], expected, f"{label} authority {field}")
        else:
            _exact_bool(value[field], expected, f"{label} authority {field}")


def _artifact_descriptor_hashes(
    request: Mapping[str, Any], labels: tuple[str, ...], label: str
) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in labels:
        descriptor = request.get(name)
        if type(descriptor) is not dict or set(descriptor) != {
            "artifact_id", "relative_path", "sha256", "size_bytes",
        }:
            raise EvidenceError(f"{label} {name} descriptor schema mismatch")
        if descriptor.get("artifact_id") != name:
            raise EvidenceError(f"{label} {name} artifact identity mismatch")
        _require_sha256(descriptor.get("sha256"), f"{label} {name} SHA-256")
        if type(descriptor.get("relative_path")) is not str \
                or not descriptor["relative_path"]:
            raise EvidenceError(f"{label} {name} relative path is invalid")
        if type(descriptor.get("size_bytes")) is not int \
                or descriptor["size_bytes"] < 0:
            raise EvidenceError(f"{label} {name} size is invalid")
        result[name] = descriptor["sha256"]
    return dict(sorted(result.items()))


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or len(value) != 64 \
            or any(character not in "0123456789abcdef" for character in value):
        raise EvidenceError(f"{label} is not a lowercase SHA-256")
    return value


def _path_entry_exists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise EvidenceError(f"path entry cannot be inspected: {path}") from error
    return True


def _legacy_frozen_artifact_set(root: Path) -> tuple[dict[str, str], str]:
    relatives = {"V34_RESULT_VALIDATION_FAILURE.json"}
    relatives.update(FAILED_CYCLE_RECEIPTS.values())
    for cycle in SEALED_CYCLES:
        relatives.add(f"evidence/orchestrator_state_v2/official_seal_v{cycle}.json")
    for result_relative, ledger_relative in LEGACY_RUN_ARTIFACTS.values():
        relatives.update((result_relative, ledger_relative))
    if len(relatives) != LEGACY_FROZEN_ARTIFACT_COUNT:
        raise EvidenceError("legacy frozen artifact path set changed")
    hashes = {
        relative: sha256_bytes(
            _read_root_relative_regular_bytes(
                root, relative, "legacy frozen artifact"
            )
        )
        for relative in sorted(relatives)
    }
    stream = b"".join(
        f"{hashes[relative]}  {relative}\n".encode("utf-8")
        for relative in sorted(hashes)
    )
    aggregate = sha256_bytes(stream)
    if aggregate != LEGACY_FROZEN_ARTIFACT_STREAM_SHA256:
        raise EvidenceError("legacy frozen artifact set hash mismatch")
    return hashes, aggregate


def _failed_cycle_output_paths(root: Path, cycle: int) -> list[str]:
    root_fd, root_identity = _open_root_anchor(root)
    try:
        opened = _open_existing_parent_at(root_fd, ("evidence", "__sentinel__"))
        if opened is None:
            raise EvidenceError("legacy evidence directory is missing")
        evidence_fd, _ = opened
        matches: list[str] = []

        def walk(directory_fd: int, prefix: tuple[str, ...]) -> None:
            for name in sorted(os.listdir(directory_fd)):
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                relative = "/".join((*prefix, name))
                if stat.S_ISDIR(info.st_mode):
                    _require_safe_directory(
                        info, f"legacy evidence directory {relative}"
                    )
                    child_fd = os.open(
                        name,
                        os.O_RDONLY
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=directory_fd,
                    )
                    try:
                        opened = _safe_directory_fd(
                            child_fd,
                            f"opened legacy evidence directory {relative}",
                        )
                        if _directory_identity(info) != _directory_identity(opened):
                            raise EvidenceError(
                                f"legacy evidence directory changed while opening: {relative}"
                            )
                        walk(child_fd, (*prefix, name))
                        after = _safe_directory_fd(
                            child_fd,
                            f"reread legacy evidence directory {relative}",
                        )
                        named = os.stat(
                            name, dir_fd=directory_fd, follow_symlinks=False
                        )
                        _require_safe_directory(
                            named, f"legacy evidence directory pathname fence {relative}"
                        )
                        if _directory_identity(after) != _directory_identity(opened) \
                                or _directory_identity(named) \
                                != _directory_identity(opened):
                            raise EvidenceError(
                                f"legacy evidence directory changed during traversal: {relative}"
                            )
                    finally:
                        os.close(child_fd)
                elif stat.S_ISREG(info.st_mode):
                    _require_safe_regular(
                        info, f"legacy evidence file {relative}"
                    )
                    stem_tokens = Path(name).stem.lower().split("_")
                    if name.lower().startswith(("result_", "proposal_ledger_")) \
                            and f"v{cycle}" in stem_tokens:
                        matches.append(relative)
                else:
                    raise EvidenceError(
                        f"unsafe legacy evidence entry while checking V{cycle}: {relative}"
                    )

        try:
            walk(evidence_fd, ("evidence",))
        finally:
            os.close(evidence_fd)
        _assert_root_identity(root, root_identity, root_fd)
        return matches
    finally:
        os.close(root_fd)


def _validate_legacy_seal(payload: Mapping[str, Any], cycle: int) -> None:
    base_keys = {
        "schema_version", "cycle_id", "official_execution_ordinal",
        "preregistration_sha256", "script_sha256", "test_sha256",
        "registry_sha256", "source_manifest_sha256", "result_file_sha256",
        "embedded_result_sha256", "ledger_sha256", "signal_id_set_sha256",
        "signals", "effective_bet_days", "recovered_without_rerun",
        "system_acceptance", "strategy_profit_gate", "authority",
        "official_seal_sha256",
    }
    recovery_keys = {
        "authorized_recovery_execution", "authorized_recovery_ordinal",
        "recovery_authorization_sha256",
    }
    expected_keys = base_keys if cycle == 25 else base_keys | recovery_keys
    if set(payload) != expected_keys:
        raise EvidenceError(f"V{cycle} legacy seal schema changed")
    if type(payload.get("schema_version")) is not int \
            or payload["schema_version"] != 2 \
            or payload.get("cycle_id") != f"V{cycle}" \
            or type(payload.get("official_execution_ordinal")) is not int \
            or payload["official_execution_ordinal"] != 1 \
            or type(payload.get("signals")) is not int or payload["signals"] <= 0 \
            or type(payload.get("effective_bet_days")) is not int \
            or payload["effective_bet_days"] <= 0 \
            or payload.get("recovered_without_rerun") is not False:
        raise EvidenceError(f"V{cycle} legacy seal execution identity is invalid")
    if cycle != 25 and (
        payload.get("authorized_recovery_execution") is not False
        or type(payload.get("authorized_recovery_ordinal")) is not int
        or payload["authorized_recovery_ordinal"] != 0
        or payload.get("recovery_authorization_sha256") is not None
    ):
        raise EvidenceError(f"V{cycle} legacy recovery identity is invalid")
    for field in (
        "preregistration_sha256", "script_sha256", "test_sha256",
        "registry_sha256", "source_manifest_sha256", "result_file_sha256",
        "embedded_result_sha256", "ledger_sha256", "signal_id_set_sha256",
        "official_seal_sha256",
    ):
        _require_sha256(payload.get(field), f"V{cycle} {field}")
    if payload["official_seal_sha256"] != embedded(payload, "official_seal_sha256"):
        raise EvidenceError(f"V{cycle} legacy seal self-hash mismatch")
    _validate_exact_authority(payload.get("authority"), f"V{cycle} legacy seal")
    if payload.get("system_acceptance") != {
        "passed": True,
        "paper_only": True,
        "external_orders": 0,
        "holdout_state": "UNOPENED",
        "restart_safe_seal": True,
    }:
        raise EvidenceError(f"V{cycle} legacy system-acceptance claim mismatch")
    profit = payload.get("strategy_profit_gate")
    if type(profit) is not dict or set(profit) != {
        "initial_equity_jpy", "normal_monthly_multiples",
        "adverse_monthly_multiples", "normal_2x_pass", "adverse_2x_pass",
        "stretch_3x_pass", "unopened_holdout_reproduced", "passed",
        "adoption_authorized",
    } or type(profit.get("initial_equity_jpy")) is not int \
            or profit["initial_equity_jpy"] != 200_000 \
            or profit.get("normal_2x_pass") is not False \
            or profit.get("adverse_2x_pass") is not False \
            or profit.get("stretch_3x_pass") is not False \
            or profit.get("unopened_holdout_reproduced") is not False \
            or profit.get("passed") is not False \
            or profit.get("adoption_authorized") is not False \
            or type(profit.get("normal_monthly_multiples")) is not dict \
            or type(profit.get("adverse_monthly_multiples")) is not dict:
        raise EvidenceError(f"V{cycle} legacy profit-gate claim mismatch")


def _legacy_signal_ids(ledger_bytes: bytes, cycle: int) -> list[str]:
    if not ledger_bytes or not ledger_bytes.endswith(b"\n"):
        raise EvidenceError(f"V{cycle} legacy ledger must be nonempty newline-delimited JSON")
    signal_ids: list[str] = []
    for line_number, line in enumerate(ledger_bytes.splitlines(), start=1):
        if not line:
            raise EvidenceError(f"V{cycle} legacy ledger contains an empty row")
        row = _strict_json_object(line, f"V{cycle} legacy ledger row {line_number}")
        signal_id = row.get("signal_id")
        if type(signal_id) is not str or not signal_id:
            raise EvidenceError(f"V{cycle} legacy ledger signal_id is invalid")
        signal_ids.append(signal_id)
    if len(signal_ids) != len(set(signal_ids)):
        raise EvidenceError(f"V{cycle} legacy ledger contains duplicate signal IDs")
    return signal_ids


def _validate_legacy_run(
    root: Path,
    cycle: int,
    *,
    expected_result_file_sha256: str,
    expected_ledger_sha256: str,
    expected_embedded_result_sha256: str | None,
    expected_signal_id_set_sha256: str | None,
    expected_signals: int | None,
    expected_effective_bet_days: int | None,
) -> dict[str, Any]:
    try:
        result_relative, ledger_relative = LEGACY_RUN_ARTIFACTS[cycle]
    except KeyError as error:
        raise EvidenceError(f"V{cycle} legacy run path is not frozen") from error
    result_bytes = _read_root_relative_regular_bytes(
        root, result_relative, f"V{cycle} legacy result"
    )
    ledger_bytes = _read_root_relative_regular_bytes(
        root, ledger_relative, f"V{cycle} legacy ledger"
    )
    result_file_sha256 = sha256_bytes(result_bytes)
    ledger_sha256 = sha256_bytes(ledger_bytes)
    if result_file_sha256 != expected_result_file_sha256:
        raise EvidenceError(f"V{cycle} legacy result file hash mismatch")
    if ledger_sha256 != expected_ledger_sha256:
        raise EvidenceError(f"V{cycle} legacy ledger file hash mismatch")
    result = _strict_json_object(result_bytes, f"V{cycle} legacy result")
    embedded_result_sha256 = _require_sha256(
        result.get("result_sha256"), f"V{cycle} embedded result"
    )
    if embedded_result_sha256 != embedded(result, "result_sha256"):
        raise EvidenceError(f"V{cycle} embedded result self-hash mismatch")
    if expected_embedded_result_sha256 is not None \
            and embedded_result_sha256 != expected_embedded_result_sha256:
        raise EvidenceError(f"V{cycle} sealed embedded result hash mismatch")
    if result.get("proposal_ledger") != ledger_relative \
            or result.get("proposal_ledger_sha256") != ledger_sha256:
        raise EvidenceError(f"V{cycle} result does not bind the exact ledger")
    result_cycle = result.get("cycle_id")
    if result_cycle is not None and result_cycle != f"V{cycle}":
        raise EvidenceError(f"V{cycle} result cycle identity mismatch")
    if result.get("external_orders") != 0 or result.get("live_authority") is not False:
        raise EvidenceError(f"V{cycle} result authority boundary mismatch")
    if result.get("authority") is not None:
        _validate_exact_authority(result.get("authority"), f"V{cycle} result")
    holdout = result.get("holdout")
    if holdout is not None and (
        type(holdout) is not dict
        or holdout.get("state") != "UNOPENED"
        or ("may_execute" in holdout and holdout.get("may_execute") is not False)
    ):
        raise EvidenceError(f"V{cycle} result holdout boundary mismatch")
    raw_signals = result.get("raw_signals")
    effective_bet_days = result.get("effective_bet_days")
    if type(raw_signals) is not int or raw_signals <= 0 \
            or type(effective_bet_days) is not int or effective_bet_days <= 0:
        raise EvidenceError(f"V{cycle} result sample identity is invalid")
    if expected_signals is not None and raw_signals != expected_signals:
        raise EvidenceError(f"V{cycle} sealed signal count mismatch")
    if expected_effective_bet_days is not None \
            and effective_bet_days != expected_effective_bet_days:
        raise EvidenceError(f"V{cycle} sealed effective-day count mismatch")
    signal_ids = _legacy_signal_ids(ledger_bytes, cycle)
    if len(signal_ids) != raw_signals:
        raise EvidenceError(f"V{cycle} legacy ledger row count mismatch")
    signal_id_set_sha256 = sha256_bytes(canonical(sorted(signal_ids)))
    if expected_signal_id_set_sha256 is not None \
            and signal_id_set_sha256 != expected_signal_id_set_sha256:
        raise EvidenceError(f"V{cycle} sealed signal-ID-set hash mismatch")
    result_signal_hash = result.get("signal_id_set_sha256")
    if result_signal_hash is not None \
            and result_signal_hash != signal_id_set_sha256:
        raise EvidenceError(f"V{cycle} result signal-ID-set hash mismatch")
    return {
        "result_path": result_relative,
        "result_file_sha256": result_file_sha256,
        "embedded_result_sha256": embedded_result_sha256,
        "ledger_path": ledger_relative,
        "ledger_sha256": ledger_sha256,
        "signal_id_set_sha256": signal_id_set_sha256,
        "signals": raw_signals,
        "effective_bet_days": effective_bet_days,
    }


def _validate_v34_failure(payload: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version", "cycle_id", "status", "official_execution_ordinal",
        "subprocess_returncode", "result_file_sha256", "ledger_file_sha256",
        "result_admitted", "official_seal_exists", "metrics_admissible",
        "profit_proven", "numeric_results_may_not_be_used_for_rule_selection",
        "failure_class", "validation_error", "validation_error_sha256",
        "structural_root_cause", "v34_rerun_permitted", "next_cycle_contract",
        "holdout", "authority",
    }
    if set(payload) != expected_keys:
        raise EvidenceError("V34 validation-failure schema changed")
    if type(payload.get("schema_version")) is not int \
            or payload["schema_version"] != 1 \
            or payload.get("cycle_id") != "V34" \
            or payload.get("status") != "FAILED_RESULT_VALIDATION_NO_RERUN" \
            or type(payload.get("official_execution_ordinal")) is not int \
            or payload["official_execution_ordinal"] != 1 \
            or type(payload.get("subprocess_returncode")) is not int \
            or payload["subprocess_returncode"] != 0 \
            or payload.get("result_admitted") is not False \
            or payload.get("official_seal_exists") is not False \
            or payload.get("metrics_admissible") is not False \
            or payload.get("profit_proven") is not False \
            or payload.get("numeric_results_may_not_be_used_for_rule_selection") \
                is not True \
            or payload.get("v34_rerun_permitted") is not False \
            or payload.get("failure_class") \
                != "PREREGISTERED_INVENTORY_CAP_CONTRADICTION":
        raise EvidenceError("V34 validation-failure state is invalid")
    for field in ("result_file_sha256", "ledger_file_sha256", "validation_error_sha256"):
        _require_sha256(payload.get(field), f"V34 {field}")
    validation_error = payload.get("validation_error")
    if type(validation_error) is not str or not validation_error \
            or sha256_bytes(validation_error.encode("utf-8")) \
                != payload["validation_error_sha256"]:
        raise EvidenceError("V34 validation error hash mismatch")
    if type(payload.get("structural_root_cause")) is not str \
            or not payload["structural_root_cause"]:
        raise EvidenceError("V34 structural root cause is missing")
    if payload.get("holdout") != {"state": "UNOPENED", "may_execute": False}:
        raise EvidenceError("V34 holdout boundary mismatch")
    _validate_exact_authority(payload.get("authority"), "V34 failure receipt")
    next_cycle = payload.get("next_cycle_contract")
    if type(next_cycle) is not dict \
            or next_cycle.get("proposed_cycle") != "V35" \
            or next_cycle.get("outcome_or_cost_used") is not False \
            or next_cycle.get("new_preregistration_required") is not True:
        raise EvidenceError("V34 successor contract is invalid")


def _validate_failed_cycle_receipt(payload: Mapping[str, Any], cycle: int) -> None:
    if payload.get("cycle_id") != f"V{cycle}":
        raise EvidenceError(f"V{cycle} failure receipt identity is invalid")
    _validate_exact_authority(payload.get("authority"), f"V{cycle} failure receipt")
    if cycle == 26:
        execution = payload.get("execution_evidence")
        strategy = payload.get("strategy_evidence")
        successor = payload.get("next_work_order")
        if payload.get("status") \
                != "FAILED_AUTHORIZED_RECOVERY_NO_RESULT_RERUN_FORBIDDEN" \
                or type(execution) is not dict \
                or execution.get("result_file_exists") is not False \
                or execution.get("ledger_file_exists") is not False \
                or execution.get("second_recovery_forbidden") is not True \
                or type(strategy) is not dict \
                or strategy.get("result_observed") is not False \
                or strategy.get("metrics_available") is not False \
                or strategy.get("profit_proven") is not False \
                or type(successor) is not dict \
                or successor.get("v26_may_not_be_replayed") is not True \
                or successor.get("holdout_must_remain_unopened") is not True:
            raise EvidenceError("V26 failure receipt semantics are invalid")
    elif cycle == 32:
        result_state = payload.get("result_state")
        if payload.get("status") != "FAILED_OFFICIAL_EXECUTION_NO_RESULT_NO_RERUN" \
                or type(result_state) is not dict \
                or result_state.get("result_file_exists") is not False \
                or result_state.get("ledger_file_exists") is not False \
                or result_state.get("metrics_available") is not False \
                or result_state.get("profit_proven") is not False \
                or result_state.get("v32_may_not_be_replayed") is not True \
                or payload.get("holdout") != {"state": "UNOPENED", "may_execute": False}:
            raise EvidenceError("V32 failure receipt semantics are invalid")
    elif cycle == 36:
        if payload.get("status") != "FAILED_OFFICIAL_EXECUTION_NO_RESULT_NO_RERUN" \
                or payload.get("result_exists") is not False \
                or payload.get("metrics_available") is not False \
                or payload.get("profit_proven") is not False \
                or payload.get("strategy_observed") is not False \
                or payload.get("rerun_permitted") is not False \
                or payload.get("holdout") != {"state": "UNOPENED", "may_execute": False}:
            raise EvidenceError("V36 failure receipt semantics are invalid")
    else:
        raise EvidenceError(f"unrecognized failure receipt cycle: V{cycle}")


def _write_all(descriptor: int, value: bytes) -> None:
    offset = 0
    while offset < len(value):
        written = os.write(descriptor, value[offset:])
        if written <= 0:
            raise EvidenceError("short evidence write")
        offset += written


def _open_root_anchor(root: Path) -> tuple[int, tuple[int, int, int, int]]:
    before = os.lstat(root)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise EvidenceError("checkpoint root must be a non-symlink directory")
    _require_safe_directory(before, "checkpoint root")
    descriptor = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    opened = _safe_directory_fd(descriptor, "opened checkpoint root")
    if _directory_identity(before) != _directory_identity(opened):
        os.close(descriptor)
        raise EvidenceError("checkpoint root changed while opening")
    try:
        named_after = os.lstat(root)
        _require_safe_directory(named_after, "checkpoint root pathname fence")
    except BaseException:
        os.close(descriptor)
        raise
    if _directory_identity(named_after) != _directory_identity(opened):
        os.close(descriptor)
        raise EvidenceError("checkpoint root pathname changed while opening")
    return descriptor, _directory_identity(opened)


def _assert_root_identity(
    root: Path,
    identity: tuple[int, int, int, int],
    descriptor: int | None = None,
) -> None:
    try:
        current = os.lstat(root)
    except OSError as error:
        raise EvidenceError("checkpoint root path is no longer available") from error
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
        raise EvidenceError("checkpoint root path identity changed")
    _require_safe_directory(current, "checkpoint root path")
    if _directory_identity(current) != identity:
        raise EvidenceError("checkpoint root path identity changed")
    if descriptor is not None:
        try:
            opened = _safe_directory_fd(
                descriptor, "checkpoint root descriptor fence"
            )
        except OSError as error:
            raise EvidenceError("checkpoint root descriptor is unavailable") from error
        if _directory_identity(opened) != identity:
            raise EvidenceError("checkpoint root descriptor identity changed")
        named_after = os.lstat(root)
        _require_safe_directory(named_after, "checkpoint root final pathname fence")
        if _directory_identity(named_after) != identity:
            raise EvidenceError("checkpoint root final pathname identity changed")


def _safe_relative(relative: str) -> tuple[str, ...]:
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or not candidate.parts \
            or any(part in {"", ".", ".."} for part in candidate.parts):
        raise EvidenceError("evidence artifact path is unsafe")
    return candidate.parts


def _open_parent_at(root_fd: int, parts: tuple[str, ...]) -> tuple[int, str]:
    descriptor = os.dup(root_fd)
    try:
        _safe_directory_fd(descriptor, "evidence directory chain root")
        for part in parts[:-1]:
            try:
                os.mkdir(part, 0o700, dir_fd=descriptor)
                os.fsync(descriptor)
            except FileExistsError:
                pass
            named = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            _require_safe_directory(named, "evidence directory chain pathname")
            child = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                child_info = _safe_directory_fd(
                    child, "opened evidence directory chain"
                )
            except EvidenceError:
                os.close(child)
                raise
            if _directory_identity(named) != _directory_identity(child_info):
                os.close(child)
                raise EvidenceError("evidence directory chain changed while opening")
            named_after = os.stat(
                part, dir_fd=descriptor, follow_symlinks=False
            )
            _require_safe_directory(
                named_after, "evidence directory chain pathname fence"
            )
            if _directory_identity(named_after) != _directory_identity(child_info):
                os.close(child)
                raise EvidenceError(
                    "evidence directory chain pathname changed while opening"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor, parts[-1]
    except Exception:
        os.close(descriptor)
        raise


def _open_existing_parent_at(
    root_fd: int, parts: tuple[str, ...]
) -> tuple[int, str] | None:
    descriptor = os.dup(root_fd)
    try:
        _safe_directory_fd(descriptor, "existing evidence directory chain root")
        for part in parts[:-1]:
            try:
                named = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
                _require_safe_directory(
                    named, "existing evidence directory chain pathname"
                )
                child = os.open(
                    part,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                os.close(descriptor)
                return None
            try:
                child_info = _safe_directory_fd(
                    child, "opened existing evidence directory chain"
                )
            except EvidenceError:
                os.close(child)
                raise
            if _directory_identity(named) != _directory_identity(child_info):
                os.close(child)
                raise EvidenceError(
                    "existing evidence directory chain changed while opening"
                )
            named_after = os.stat(
                part, dir_fd=descriptor, follow_symlinks=False
            )
            _require_safe_directory(
                named_after, "existing evidence directory chain pathname fence"
            )
            if _directory_identity(named_after) != _directory_identity(child_info):
                os.close(child)
                raise EvidenceError(
                    "existing evidence directory chain pathname changed while opening"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor, parts[-1]
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_at(
    parent_fd: int,
    name: str,
    *,
    maximum_bytes: int = MAX_ARTIFACT_BYTES,
) -> bytes:
    _safe_directory_fd(parent_fd, "checkpoint evidence parent")
    if type(maximum_bytes) is not int or maximum_bytes < 0:
        raise EvidenceError("checkpoint evidence read bound is invalid")
    named_before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    _require_safe_regular(named_before, "checkpoint evidence pathname")
    descriptor = os.open(
        name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd
    )
    try:
        info = _safe_regular_fd(descriptor, "opened checkpoint evidence")
        if _regular_identity(named_before) != _regular_identity(info):
            raise EvidenceError("checkpoint evidence changed while opening")
        if info.st_size < 0 or info.st_size > maximum_bytes:
            raise EvidenceError("checkpoint evidence exceeds its byte bound")
        identity = _regular_identity(info)
        chunks: list[bytes] = []
        offset = 0
        while offset < info.st_size:
            chunk = os.pread(descriptor, min(1024 * 1024, info.st_size - offset), offset)
            if not chunk:
                raise EvidenceError("checkpoint evidence was truncated while reading")
            chunks.append(chunk)
            offset += len(chunk)
        after = _safe_regular_fd(descriptor, "reread checkpoint evidence")
        after_identity = _regular_identity(after)
        if os.pread(descriptor, 1, info.st_size) or after_identity != identity:
            raise EvidenceError("checkpoint evidence changed while reading")
        try:
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as error:
            raise EvidenceError(
                "checkpoint evidence pathname changed while reading"
            ) from error
        _require_safe_regular(named, "reread checkpoint evidence pathname")
        named_identity = _regular_identity(named)
        if named_identity != identity:
            raise EvidenceError("checkpoint evidence pathname changed while reading")
        _safe_directory_fd(parent_fd, "checkpoint evidence parent final fence")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _cleanup_builder_partials(parent_fd: int, final_name: str) -> None:
    _safe_directory_fd(parent_fd, "builder partial parent")
    prefix = f".{final_name}.builder-partial-"
    for name in os.listdir(parent_fd):
        if not name.startswith(prefix):
            continue
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() \
                or info.st_mode & 0o022 or info.st_nlink not in {1, 2}:
            raise EvidenceError("builder partial pathname is unsafe")
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            opened = _safe_regular_fd(
                descriptor,
                "opened builder partial",
                allowed_nlinks=frozenset({info.st_nlink}),
            )
            if _regular_identity(info) != _regular_identity(opened):
                raise EvidenceError("builder partial changed while opening")
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if _regular_identity(named) != _regular_identity(opened):
                raise EvidenceError("builder partial pathname changed before cleanup")
            if opened.st_nlink == 2:
                try:
                    final = os.stat(
                        final_name, dir_fd=parent_fd, follow_symlinks=False
                    )
                except FileNotFoundError as error:
                    raise EvidenceError(
                        "two-link builder partial lacks its final artifact"
                    ) from error
                if _regular_identity(final) != _regular_identity(opened):
                    raise EvidenceError(
                        "two-link builder partial is not bound to its final artifact"
                    )
            os.unlink(name, dir_fd=parent_fd)
            try:
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise EvidenceError("builder partial remained after cleanup")
            opened_after_unlink = _safe_regular_fd(
                descriptor,
                "unlinked builder partial",
                allowed_nlinks=frozenset({0, 1}),
            )
            expected_links = 1 if info.st_nlink == 2 else 0
            if opened_after_unlink.st_nlink != expected_links \
                    or (opened_after_unlink.st_dev, opened_after_unlink.st_ino) \
                    != (opened.st_dev, opened.st_ino):
                raise EvidenceError(
                    "builder partial link count changed during cleanup"
                )
            if info.st_nlink == 2:
                final_after = os.stat(
                    final_name, dir_fd=parent_fd, follow_symlinks=False
                )
                _require_safe_regular(
                    final_after, "recovered builder final artifact"
                )
                if (final_after.st_dev, final_after.st_ino) \
                        != (opened.st_dev, opened.st_ino):
                    raise EvidenceError("recovered builder final artifact changed")
            os.fsync(parent_fd)
            _safe_directory_fd(parent_fd, "builder partial parent final fence")
        finally:
            os.close(descriptor)


def _cleanup_all_builder_partials(root_fd: int, relatives: set[str]) -> None:
    """Recover only builder-owned temp links before exact file-set validation."""
    for relative in sorted(relatives):
        parent_fd, name = _open_parent_at(root_fd, _safe_relative(relative))
        try:
            _cleanup_builder_partials(parent_fd, name)
        finally:
            os.close(parent_fd)


def _cleanup_existing_builder_partials_at(root_fd: int, relative: str) -> None:
    opened = _open_existing_parent_at(root_fd, _safe_relative(relative))
    if opened is None:
        return
    parent_fd, name = opened
    try:
        _cleanup_builder_partials(parent_fd, name)
    finally:
        os.close(parent_fd)


def atomic_bytes_at(root_fd: int, relative: str, value: bytes) -> None:
    parent_fd, name = _open_parent_at(root_fd, _safe_relative(relative))
    temporary = f".{name}.builder-partial-{secrets.token_hex(12)}"
    descriptor = -1
    created = False
    linked = False
    created_inode: tuple[int, int] | None = None
    try:
        _safe_directory_fd(parent_fd, "builder publication parent")
        _cleanup_builder_partials(parent_fd, name)
        try:
            existing = _read_regular_at(
                parent_fd, name, maximum_bytes=len(value)
            )
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing != value:
                raise EvidenceError(f"refusing to overwrite mismatched evidence: {relative}")
            return
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        created = True
        created_info = _safe_regular_fd(descriptor, "new builder partial")
        created_inode = (created_info.st_dev, created_info.st_ino)
        _write_all(descriptor, value)
        os.fsync(descriptor)
        written = _safe_regular_fd(descriptor, "written builder partial")
        temporary_named = os.stat(
            temporary, dir_fd=parent_fd, follow_symlinks=False
        )
        _require_safe_regular(temporary_named, "builder partial pathname fence")
        if _regular_identity(written) != _regular_identity(temporary_named):
            raise EvidenceError("builder partial changed before publication")
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            linked = True
        except FileExistsError:
            if _read_regular_at(
                parent_fd, name, maximum_bytes=len(value)
            ) != value:
                raise EvidenceError(f"evidence publication collision: {relative}")
        if linked:
            linked_fd = _safe_regular_fd(
                descriptor,
                "linked builder publication",
                allowed_nlinks=frozenset({2}),
            )
            linked_temp = os.stat(
                temporary, dir_fd=parent_fd, follow_symlinks=False
            )
            linked_final = os.stat(
                name, dir_fd=parent_fd, follow_symlinks=False
            )
            for info in (linked_fd, linked_temp, linked_final):
                if not stat.S_ISREG(info.st_mode) \
                        or info.st_uid != os.geteuid() \
                        or info.st_mode & 0o022 or info.st_nlink != 2:
                    raise EvidenceError("linked builder publication is unsafe")
            if not (
                _regular_identity(linked_fd)
                == _regular_identity(linked_temp)
                == _regular_identity(linked_final)
            ):
                raise EvidenceError("builder publication link identity changed")
        os.fsync(parent_fd)
        os.unlink(temporary, dir_fd=parent_fd)
        created = False
        os.fsync(parent_fd)
        try:
            os.stat(temporary, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise EvidenceError("builder partial remained after publication")
        after_unlink = _safe_regular_fd(
            descriptor,
            "published builder artifact descriptor fence",
            allowed_nlinks=frozenset({0, 1}),
        )
        expected_links = 1 if linked else 0
        if after_unlink.st_nlink != expected_links \
                or (after_unlink.st_dev, after_unlink.st_ino) != created_inode:
            raise EvidenceError("builder partial link count changed unexpectedly")
        if linked:
            final_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            _require_safe_regular(final_after, "published builder artifact")
            if (final_after.st_dev, final_after.st_ino) != created_inode:
                raise EvidenceError("published builder artifact identity changed")
        _safe_directory_fd(parent_fd, "builder publication parent final fence")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if created:
            try:
                partial = os.stat(
                    temporary, dir_fd=parent_fd, follow_symlinks=False
                )
                if stat.S_ISREG(partial.st_mode) \
                        and partial.st_uid == os.geteuid() \
                        and not partial.st_mode & 0o022 \
                        and partial.st_nlink in {1, 2} \
                        and (partial.st_dev, partial.st_ino) == created_inode:
                    os.unlink(temporary, dir_fd=parent_fd)
                    os.fsync(parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _acquire_builder_lock(root_fd: int) -> int:
    _safe_directory_fd(root_fd, "builder lock root")
    try:
        named_before = os.stat(
            BUILDER_LOCK_PATH, dir_fd=root_fd, follow_symlinks=False
        )
    except FileNotFoundError as error:
        raise EvidenceError("reviewed persistent builder lock is missing") from error
    _require_safe_regular(named_before, "builder lock pathname")
    try:
        descriptor = os.open(
            BUILDER_LOCK_PATH,
            os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
    except FileNotFoundError as error:
        raise EvidenceError("reviewed persistent builder lock is missing") from error
    try:
        info = _safe_regular_fd(descriptor, "opened builder lock")
        named_after = os.stat(
            BUILDER_LOCK_PATH, dir_fd=root_fd, follow_symlinks=False
        )
        _require_safe_regular(named_after, "builder lock pathname fence")
    except BaseException:
        os.close(descriptor)
        raise
    if _regular_identity(named_before) != _regular_identity(info) \
            or _regular_identity(named_after) != _regular_identity(info):
        os.close(descriptor)
        raise EvidenceError("builder lock changed while opening")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise EvidenceError("another checkpoint builder owns the lock") from error
    _safe_directory_fd(root_fd, "builder lock root final fence")
    return descriptor


def _assert_builder_lock_identity(root_fd: int, lock_fd: int) -> None:
    _safe_directory_fd(root_fd, "builder lock root identity fence")
    try:
        held = _safe_regular_fd(lock_fd, "builder lock identity fence")
        named = os.stat(
            BUILDER_LOCK_PATH,
            dir_fd=root_fd,
            follow_symlinks=False,
        )
        _require_safe_regular(named, "builder lock identity fence")
    except (OSError, EvidenceError) as error:
        raise EvidenceError("builder lock pathname identity changed") from error
    if _regular_identity(held) != _regular_identity(named):
        raise EvidenceError("builder lock pathname identity changed")


def _read_optional_artifact_at(root_fd: int, relative: str) -> bytes | None:
    opened = _open_existing_parent_at(root_fd, _safe_relative(relative))
    if opened is None:
        return None
    parent_fd, name = opened
    try:
        try:
            return _read_regular_at(
                parent_fd, name, maximum_bytes=MAX_JSON_BYTES
            )
        except FileNotFoundError:
            return None
    finally:
        os.close(parent_fd)


def _read_root_relative_regular_bytes(
    root: Path,
    relative: str,
    label: str,
    *,
    maximum_bytes: int = MAX_ARTIFACT_BYTES,
) -> bytes:
    root_fd, root_identity = _open_root_anchor(root)
    try:
        opened = _open_existing_parent_at(root_fd, _safe_relative(relative))
        if opened is None:
            raise EvidenceError(f"{label} is missing: {relative}")
        parent_fd, name = opened
        try:
            try:
                value = _read_regular_at(
                    parent_fd, name, maximum_bytes=maximum_bytes
                )
            except FileNotFoundError as error:
                raise EvidenceError(f"{label} is missing: {relative}") from error
        finally:
            os.close(parent_fd)
        _assert_root_identity(root, root_identity, root_fd)
        return value
    finally:
        os.close(root_fd)


def _invalidate_terminal_commit_at(root_fd: int, expected_bytes: bytes) -> None:
    opened = _open_existing_parent_at(
        root_fd, _safe_relative(CHECKPOINT_COMMIT_PATH)
    )
    if opened is None:
        return
    parent_fd, name = opened
    descriptor = -1
    try:
        _safe_directory_fd(parent_fd, "terminal commit parent")
        try:
            named_before = os.stat(
                name, dir_fd=parent_fd, follow_symlinks=False
            )
            _require_safe_regular(
                named_before, "terminal commit invalidation pathname"
            )
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            return
        info = _safe_regular_fd(
            descriptor, "opened terminal commit invalidation target"
        )
        identity = _regular_identity(info)
        if _regular_identity(named_before) != identity:
            raise EvidenceError("terminal commit changed while opening for invalidation")
        if info.st_size < 0 or info.st_size > MAX_JSON_BYTES:
            raise EvidenceError("terminal commit exceeds its byte bound")
        chunks: list[bytes] = []
        offset = 0
        while offset < info.st_size:
            chunk = os.pread(descriptor, min(1024 * 1024, info.st_size - offset), offset)
            if not chunk:
                raise EvidenceError("terminal commit truncated during invalidation")
            chunks.append(chunk)
            offset += len(chunk)
        after = _safe_regular_fd(
            descriptor, "reread terminal commit invalidation target"
        )
        if _regular_identity(after) != identity:
            raise EvidenceError("terminal commit changed during invalidation read")
        if os.pread(descriptor, 1, info.st_size) or b"".join(chunks) != expected_bytes:
            raise EvidenceError("terminal commit bytes differ from expected invalidation target")
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        _require_safe_regular(named, "terminal commit invalidation pathname fence")
        if _regular_identity(named) != identity:
            raise EvidenceError("terminal commit identity changed before invalidation")
        os.unlink(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise EvidenceError("terminal commit remained after invalidation")
        unlinked = _safe_regular_fd(
            descriptor,
            "unlinked terminal commit invalidation target",
            allowed_nlinks=frozenset({0}),
        )
        if unlinked.st_nlink != 0 \
                or (unlinked.st_dev, unlinked.st_ino) != (info.st_dev, info.st_ino):
            raise EvidenceError("terminal commit link count changed during invalidation")
        _safe_directory_fd(parent_fd, "terminal commit parent final fence")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _assert_private_directory(path: Path, label: str) -> None:
    try:
        before = os.lstat(path)
    except OSError as error:
        raise EvidenceError(f"{label} cannot be inspected") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode) \
            or before.st_uid != os.geteuid() \
            or stat.S_IMODE(before.st_mode) != 0o700:
        raise EvidenceError(f"{label} is not an owner-private directory")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = _safe_directory_fd(descriptor, label)
        if _directory_identity(before) != _directory_identity(opened):
            raise EvidenceError(f"{label} changed while opening")
    finally:
        os.close(descriptor)


def _mkdir_private(path: Path, label: str) -> None:
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    _assert_private_directory(path, label)


def _assert_distinct_private_roots(
    roots: Mapping[str, Path],
) -> dict[str, tuple[int, int]]:
    identities: dict[str, tuple[int, int]] = {}
    for label, path in roots.items():
        _assert_private_directory(path, label)
        info = os.lstat(path)
        identity = (info.st_dev, info.st_ino)
        if identity in identities.values():
            raise EvidenceError("runtime roots are not four distinct directory inodes")
        identities[label] = identity
    return identities


def _write_private_bytes(path: Path, value: bytes, label: str) -> Path:
    if type(value) is not bytes:
        raise EvidenceError(f"{label} bytes are invalid")
    _assert_private_directory(path.parent, f"{label} parent")
    # Retain a separate directory anchor so the create, reread, and fsync bind
    # to the same parent even if a pathname is replaced concurrently.
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    try:
        _safe_directory_fd(parent_fd, f"{label} parent descriptor")
        descriptor = os.open(
            path.name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        created = _safe_regular_fd(descriptor, f"created {label}")
        if not stat.S_ISREG(created.st_mode) or created.st_nlink != 1 \
                or created.st_uid != os.geteuid() \
                or stat.S_IMODE(created.st_mode) != 0o600:
            raise EvidenceError(f"{label} was not created as a private file")
        _write_all(descriptor, value)
        os.fsync(descriptor)
        reread = bytearray()
        offset = 0
        while offset < len(value):
            chunk = os.pread(descriptor, min(1024 * 1024, len(value) - offset), offset)
            if not chunk:
                raise EvidenceError(f"{label} truncated during fd reread")
            reread.extend(chunk)
            offset += len(chunk)
        if bytes(reread) != value or os.pread(descriptor, 1, len(value)):
            raise EvidenceError(f"{label} fd reread binding mismatch")
        after = _safe_regular_fd(descriptor, f"written {label}")
        if (created.st_dev, created.st_ino, created.st_nlink) != (
            after.st_dev, after.st_ino, after.st_nlink
        ) or after.st_size != len(value):
            raise EvidenceError(f"{label} changed during private write")
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if _regular_identity(after) != _regular_identity(named):
            raise EvidenceError(f"{label} pathname changed after private write")
        os.fsync(parent_fd)
        _safe_directory_fd(parent_fd, f"{label} parent final fence")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)
    if read_regular_bytes(
        path, label, maximum_bytes=len(value)
    ) != value:
        raise EvidenceError(f"{label} reopen binding mismatch")
    return path


def _copy_private_regular(
    source: Path,
    target: Path,
    label: str,
    *,
    maximum_bytes: int,
) -> Path:
    source_bytes = read_regular_bytes(
        source, f"{label} source", maximum_bytes=maximum_bytes
    )
    source_info = os.lstat(source)
    _write_private_bytes(target, source_bytes, f"{label} copy")
    target_info = os.lstat(target)
    if (source_info.st_dev, source_info.st_ino) == (
        target_info.st_dev, target_info.st_ino
    ) or source_info.st_nlink != 1 or target_info.st_nlink != 1:
        raise EvidenceError(f"{label} copy is hard-linked or externally aliased")
    if read_regular_bytes(
        source, f"{label} source fence", maximum_bytes=maximum_bytes
    ) != read_regular_bytes(
        target, f"{label} destination fence", maximum_bytes=maximum_bytes
    ):
        raise EvidenceError(f"{label} copy bytes changed across fd reread fence")
    return target


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    _write_private_bytes(path, canonical(payload) + b"\n", path.name)
    return path


def artifact(root: Path, path: Path, label: str) -> dict[str, Any]:
    maximum_bytes = (
        MAX_ARTIFACT_BYTES
        if label in {"source_blob", "oracle_ledger"}
        else MAX_CODE_BYTES
        if label in {"oracle_code_snapshot", "reference_code_snapshot"}
        else MAX_JSON_BYTES
    )
    data = read_regular_bytes(
        path, f"artifact {label}", maximum_bytes=maximum_bytes
    )
    return {
        "artifact_id": label,
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": sha256_bytes(data),
        "size_bytes": len(data),
    }


def fixture(
    root: Path, golden: Mapping[str, Any]
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    _mkdir_private(root, "Oracle input root")
    if any(root.iterdir()):
        raise EvidenceError("Oracle input root is not empty")
    inputs = golden["inputs"]
    blob_path = root / "inputs" / "source_blob.jsonl"
    _mkdir_private(blob_path.parent, "Oracle input directory")
    _write_private_bytes(
        blob_path,
        inputs["source_blob_utf8"].encode("utf-8"),
        "Oracle source blob",
    )
    payloads = {
        label: inputs[label]
        for label in (
            "source_manifest",
            "proposal",
            "execution_policy",
            "inventory_policy",
            "accounting_policy",
            "evaluation_policy",
            "instrument_registry",
            "authority_policy",
        )
    }
    paths = {
        label: write_json(root / "inputs" / f"{label}.json", payload)
        for label, payload in payloads.items()
    }
    request = {
        "schema_version": 2,
        "source_blob": artifact(root, blob_path, "source_blob"),
        "source_manifest": artifact(
            root, paths["source_manifest"], "source_manifest"
        ),
        "proposal": artifact(root, paths["proposal"], "proposal"),
        "execution_policy": artifact(root, paths["execution_policy"], "execution_policy"),
        "inventory_policy": artifact(root, paths["inventory_policy"], "inventory_policy"),
        "accounting_policy": artifact(root, paths["accounting_policy"], "accounting_policy"),
        "evaluation_policy": artifact(root, paths["evaluation_policy"], "evaluation_policy"),
        "instrument_registry": artifact(
            root, paths["instrument_registry"], "instrument_registry"
        ),
        "authority_policy": artifact(root, paths["authority_policy"], "authority_policy"),
        "output_directory": "oracle_output",
    }
    request_path = write_json(root / "inputs" / "oracle_request.json", request)
    return request, request_path, golden["expected"]


def _open_readonly_regular(path: Path) -> int:
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise EvidenceError(f"runtime artifact is not a regular non-symlink file: {path.name}")
    _require_safe_regular(before, f"runtime artifact pathname {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        after = _safe_regular_fd(
            descriptor, f"opened runtime artifact {path.name}"
        )
    except BaseException:
        os.close(descriptor)
        raise
    if _regular_identity(before) != _regular_identity(after):
        os.close(descriptor)
        raise EvidenceError(
            f"runtime artifact changed or is externally aliased: {path.name}"
        )
    try:
        named_after = os.lstat(path)
        _require_safe_regular(
            named_after, f"runtime artifact pathname fence {path.name}"
        )
    except BaseException:
        os.close(descriptor)
        raise
    if _regular_identity(named_after) != _regular_identity(after):
        os.close(descriptor)
        raise EvidenceError(f"runtime artifact pathname changed: {path.name}")
    _safe_regular_fd(descriptor, f"runtime artifact final fence {path.name}")
    return descriptor


def _open_root_relative_readonly_regular(
    root_fd: int,
    relative: str,
    label: str,
) -> int:
    opened_parent = _open_existing_parent_at(root_fd, _safe_relative(relative))
    if opened_parent is None:
        raise EvidenceError(f"{label} parent is missing")
    parent_fd, name = opened_parent
    descriptor = -1
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        _require_safe_regular(before, f"{label} pathname")
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        opened = _safe_regular_fd(descriptor, f"opened {label}")
        named_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        _require_safe_regular(named_after, f"{label} pathname fence")
        if _regular_identity(before) != _regular_identity(opened) \
                or _regular_identity(named_after) != _regular_identity(opened):
            raise EvidenceError(f"{label} changed while opening")
        _safe_regular_fd(descriptor, f"{label} final descriptor fence")
        result = descriptor
        descriptor = -1
        return result
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _fixed_bootstrap_from_launcher(
    launcher_bytes: bytes, expected_launcher_sha256: str
) -> tuple[str, str]:
    if type(launcher_bytes) is not bytes or len(launcher_bytes) > MAX_CODE_BYTES:
        raise EvidenceError("sealed launcher byte snapshot is invalid")
    if sha256_bytes(launcher_bytes) != expected_launcher_sha256:
        raise EvidenceError("launcher changed after source freeze")
    try:
        tree = ast.parse(launcher_bytes, filename=LAUNCHER_PATH)
    except SyntaxError as error:
        raise EvidenceError("sealed launcher cannot be parsed") from error
    values: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "FIXED_BOOTSTRAP_SOURCE":
            if not isinstance(node.value, ast.Constant) or type(node.value.value) is not str:
                raise EvidenceError("launcher bootstrap is not a fixed source literal")
            values.append(node.value.value)
    if len(values) != 1:
        raise EvidenceError("launcher must contain one fixed bootstrap source literal")
    bootstrap_source = values[0]
    bootstrap_sha256 = sha256_bytes(bootstrap_source.encode("utf-8"))
    if not bootstrap_source or len(bootstrap_source.encode("utf-8")) > 64 * 1024:
        raise EvidenceError("launcher bootstrap source exceeds its fixed bound")
    return bootstrap_source, bootstrap_sha256


def _invoke_launcher(
    source_root: Path,
    input_root: Path,
    output_root: Path,
    operation: str,
    request_path: Path,
    *,
    expected_launcher_sha256: str,
    trusted_oracle_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    if operation not in {"ORACLE", "VERIFIER"}:
        raise EvidenceError("unsupported sealed launcher operation")
    _assert_distinct_private_roots({
        f"{operation.lower()} input root": input_root,
        f"{operation.lower()} output root": output_root,
    })
    paths: dict[str, Path] = {
        "launcher": source_root / LAUNCHER_PATH,
        "request": request_path,
        "code": source_root / (ORACLE_PATH if operation == "ORACLE" else VERIFIER_PATH),
        "schema": source_root / (
            SCHEMA_PATH if operation == "ORACLE" else VERIFIER_SCHEMA_PATH
        ),
    }
    if operation == "ORACLE":
        paths["contract"] = source_root / CONTRACT_PATH
    else:
        if trusted_oracle_paths is None:
            raise EvidenceError("verifier requires trusted Oracle release paths")
        paths.update({
            "oracle_code": trusted_oracle_paths["code"],
            "oracle_contract": trusted_oracle_paths["contract"],
            "oracle_schema": trusted_oracle_paths["schema"],
            "reference_code": trusted_oracle_paths["reference_code"],
            "reference_contract": trusted_oracle_paths["reference_contract"],
        })

    source_relatives = {
        "launcher": LAUNCHER_PATH,
        "code": ORACLE_PATH if operation == "ORACLE" else VERIFIER_PATH,
        "schema": SCHEMA_PATH if operation == "ORACLE" else VERIFIER_SCHEMA_PATH,
    }
    if operation == "ORACLE":
        source_relatives["contract"] = CONTRACT_PATH

    source_root_fd, source_root_identity = _open_root_anchor(source_root)
    descriptors: dict[str, int] = {}
    try:
        for label, path in paths.items():
            if label in source_relatives:
                descriptors[label] = _open_root_relative_readonly_regular(
                    source_root_fd,
                    source_relatives[label],
                    f"sealed {label}",
                )
            else:
                descriptors[label] = _open_readonly_regular(path)
        descriptors["input_root"] = os.open(
            input_root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        descriptors["output_root"] = os.open(
            output_root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        _safe_directory_fd(descriptors["input_root"], "launcher input root")
        _safe_directory_fd(descriptors["output_root"], "launcher output root")
        launcher_info = _safe_regular_fd(
            descriptors["launcher"], "sealed launcher bootstrap snapshot"
        )
        launcher_bytes = b"".join(
            os.pread(
                descriptors["launcher"],
                min(1024 * 1024, launcher_info.st_size - offset),
                offset,
            )
            for offset in range(0, launcher_info.st_size, 1024 * 1024)
        )
        if len(launcher_bytes) != launcher_info.st_size \
                or os.pread(descriptors["launcher"], 1, launcher_info.st_size):
            raise EvidenceError("sealed launcher changed while snapshotting bootstrap")
        launcher_after = _safe_regular_fd(
            descriptors["launcher"], "sealed launcher bootstrap snapshot fence"
        )
        if _regular_identity(launcher_after) != _regular_identity(launcher_info):
            raise EvidenceError("sealed launcher changed while snapshotting bootstrap")
        bootstrap_source, bootstrap_source_sha256 = _fixed_bootstrap_from_launcher(
            launcher_bytes, expected_launcher_sha256
        )
        arguments = [
            str(PYTHON),
            "-I",
            "-S",
            "-B",
            "-c",
            bootstrap_source,
            "--launcher-fd",
            str(descriptors["launcher"]),
            "--expected-launcher-sha256",
            expected_launcher_sha256,
            "--bootstrap-source-sha256",
            bootstrap_source_sha256,
            "--operation",
            operation,
            "--request-fd",
            str(descriptors["request"]),
            "--input-root-fd",
            str(descriptors["input_root"]),
            "--output-root-fd",
            str(descriptors["output_root"]),
            "--code-fd",
            str(descriptors["code"]),
            "--schema-fd",
            str(descriptors["schema"]),
        ]
        for label, option in (
            ("contract", "--contract-fd"),
            ("oracle_code", "--oracle-code-fd"),
            ("oracle_contract", "--oracle-contract-fd"),
            ("oracle_schema", "--oracle-schema-fd"),
            ("reference_code", "--reference-code-fd"),
            ("reference_contract", "--reference-contract-fd"),
        ):
            if label in descriptors:
                arguments.extend((option, str(descriptors[label])))
        try:
            _assert_root_identity(
                source_root, source_root_identity, source_root_fd
            )
            completed = subprocess.run(
                arguments,
                cwd=input_root,
                env={
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "LANG": "C.UTF-8",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                pass_fds=tuple(descriptors.values()),
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired as error:
            raise EvidenceError(
                f"sealed {operation.lower()} launcher timed out"
            ) from error
        _assert_root_identity(source_root, source_root_identity, source_root_fd)
        _safe_directory_fd(
            descriptors["input_root"], "launcher input root final fence"
        )
        _safe_directory_fd(
            descriptors["output_root"], "launcher output root final fence"
        )
    finally:
        for descriptor in reversed(tuple(descriptors.values())):
            os.close(descriptor)
        os.close(source_root_fd)
    if completed.returncode != 0:
        raise EvidenceError(
            f"sealed {operation.lower()} launcher failed: {completed.stdout.strip()}"
        )
    if completed.stderr:
        raise EvidenceError(f"sealed {operation.lower()} launcher wrote stderr")
    if not completed.stdout.endswith("\n"):
        raise EvidenceError(f"sealed {operation.lower()} launcher response lacks newline")
    lines = completed.stdout.splitlines()
    if len(lines) != 1:
        raise EvidenceError(f"sealed {operation.lower()} launcher response is ambiguous")
    expected_keys = {
        "ok", "operation", "output_directory", "launcher_sha256",
        "bootstrap_attestation_sha256", "caller_asserted_bootstrap_source_sha256",
        "bootstrap_provenance", "pre_audit_capability_absence_proven",
        "interpreter_executable_sha256", "interpreter_identity_sha256",
        "interpreter_flags_sha256", "sys_path_sha256",
        "release_evidence_eligible", "local_reproducible_only",
        "outer_launch_provenance_status", "runtime_environment_scope",
        "snapshot_mode",
    }
    response = _canonical_json_object(
        (lines[0] + "\n").encode("utf-8"),
        f"sealed {operation.lower()} launcher response",
        expected_keys,
    )
    if response.get("ok") is not True \
            or response.get("operation") != operation \
            or response.get("output_directory") != (
                "oracle_output" if operation == "ORACLE" else "verifier_output"
            ) \
            or response.get("launcher_sha256") != expected_launcher_sha256 \
            or response.get("caller_asserted_bootstrap_source_sha256") \
                != bootstrap_source_sha256 \
            or response.get("bootstrap_provenance") \
                != "PYTHON_C_NOT_SELF_AUTHENTICATING" \
            or response.get("pre_audit_capability_absence_proven") is not False \
            or response.get("release_evidence_eligible") is not False \
            or response.get("local_reproducible_only") is not True \
            or response.get("outer_launch_provenance_status") \
                != "CALLER_ASSERTED_PENDING_EXTERNAL_ANCHOR" \
            or response.get("runtime_environment_scope") \
                != "HOST_LOCAL_NON_HERMETIC_STDLIB_NOT_BYTE_PINNED" \
            or response.get("snapshot_mode") != "SEALED_FD_COMPILE_EXEC_V2":
        raise EvidenceError(f"sealed {operation.lower()} launcher response mismatch")
    for field in (
        "bootstrap_attestation_sha256", "interpreter_executable_sha256",
        "interpreter_identity_sha256", "interpreter_flags_sha256", "sys_path_sha256",
    ):
        value = response.get(field)
        if type(value) is not str or len(value) != 64 \
                or any(character not in "0123456789abcdef" for character in value):
            raise EvidenceError(f"sealed launcher {field} is invalid")
    return response


def _verifier_request(
    source_root: Path,
    oracle_input_root: Path,
    oracle_output_root: Path,
    verifier_input_root: Path,
    oracle_request: Mapping[str, Any],
) -> tuple[Path, dict[str, Path]]:
    _assert_distinct_private_roots({
        "Oracle input root": oracle_input_root,
        "Oracle output root": oracle_output_root,
        "verifier input root": verifier_input_root,
    })
    if any(verifier_input_root.iterdir()):
        raise EvidenceError("verifier input root is not empty")
    input_dir = verifier_input_root / "inputs"
    copied_oracle_output_dir = verifier_input_root / "oracle_output"
    _mkdir_private(input_dir, "verifier input directory")
    _mkdir_private(copied_oracle_output_dir, "verifier Oracle output copy directory")

    for label in ORACLE_INPUT_LABELS:
        relative = oracle_request[label]["relative_path"]
        parts = _safe_relative(relative)
        source = oracle_input_root.joinpath(*parts)
        target = verifier_input_root.joinpath(*parts)
        maximum_bytes = (
            MAX_ARTIFACT_BYTES if label == "source_blob" else MAX_JSON_BYTES
        )
        _copy_private_regular(
            source,
            target,
            f"verifier {label}",
            maximum_bytes=maximum_bytes,
        )
    _copy_private_regular(
        oracle_input_root / "inputs/oracle_request.json",
        input_dir / "oracle_request.json",
        "verifier Oracle request",
        maximum_bytes=MAX_JSON_BYTES,
    )

    snapshot_paths: dict[str, Path] = {}
    for label, relative in (
        ("oracle_code_snapshot", ORACLE_PATH),
        ("oracle_contract_snapshot", CONTRACT_PATH),
        ("oracle_schema_snapshot", SCHEMA_PATH),
        ("reference_code_snapshot", REFERENCE_PATH),
        ("reference_contract_snapshot", REFERENCE_CONTRACT_PATH),
    ):
        maximum_bytes = (
            MAX_CODE_BYTES
            if label in {"oracle_code_snapshot", "reference_code_snapshot"}
            else MAX_JSON_BYTES
        )
        source_bytes = _read_root_relative_regular_bytes(
            source_root,
            relative,
            f"Oracle {label}",
            maximum_bytes=maximum_bytes,
        )
        target = input_dir / f"{label}{Path(relative).suffix}"
        _write_private_bytes(
            target, source_bytes, f"Oracle {label} snapshot"
        )
        snapshot_paths[label] = target
    copied_oracle_outputs: dict[str, Path] = {}
    for name in sorted(ORACLE_OUTPUT_FILES):
        source = oracle_output_root / "oracle_output" / name
        target = copied_oracle_output_dir / name
        copied_oracle_outputs[name] = _copy_private_regular(
            source,
            target,
            f"verifier Oracle output {name}",
            maximum_bytes=(
                MAX_ARTIFACT_BYTES if name == "oracle_ledger.jsonl" else MAX_JSON_BYTES
            ),
        )
    request = {
        "schema_version": 2,
        **{
            label: oracle_request[label]
            for label in (
                "source_blob",
                "source_manifest",
                "proposal",
                "execution_policy",
                "inventory_policy",
                "accounting_policy",
                "evaluation_policy",
                "instrument_registry",
                "authority_policy",
            )
        },
        "oracle_request": artifact(
            verifier_input_root,
            input_dir / "oracle_request.json",
            "oracle_request",
        ),
        **{
            label: artifact(verifier_input_root, path, label)
            for label, path in snapshot_paths.items()
        },
        "oracle_intent": artifact(
            verifier_input_root,
            copied_oracle_outputs["intent.json"],
            "oracle_intent",
        ),
        "oracle_commit": artifact(
            verifier_input_root,
            copied_oracle_outputs["COMMIT.json"],
            "oracle_commit",
        ),
        "oracle_ledger": artifact(
            verifier_input_root,
            copied_oracle_outputs["oracle_ledger.jsonl"],
            "oracle_ledger",
        ),
        "oracle_manifest": artifact(
            verifier_input_root,
            copied_oracle_outputs["oracle_manifest.json"],
            "oracle_manifest",
        ),
        "output_directory": "verifier_output",
    }
    request_path = write_json(input_dir / "verifier_request.json", request)
    return request_path, {
        "code": snapshot_paths["oracle_code_snapshot"],
        "contract": snapshot_paths["oracle_contract_snapshot"],
        "schema": snapshot_paths["oracle_schema_snapshot"],
        "reference_code": snapshot_paths["reference_code_snapshot"],
        "reference_contract": snapshot_paths["reference_contract_snapshot"],
    }


def legacy_coverage(root: Path) -> dict[str, Any]:
    frozen_artifact_hashes, frozen_artifact_set_sha256 = (
        _legacy_frozen_artifact_set(root)
    )
    rows = []
    missing_inputs = [
        "EXACT_EVENT_BBO_BYTES",
        "EVENT_ARRIVAL_TIMESTAMPS",
        "BASE_MICROUNITS",
        "SIGN_AWARE_CONVERSION_RECEIPTS",
        "MARGIN_GRID",
    ]
    for cycle in range(25, 42):
        run_binding: dict[str, Any] | None = None
        failure_receipt_binding: dict[str, Any] | None = None
        relative = Path(f"evidence/orchestrator_state_v2/official_seal_v{cycle}.json")
        absolute = root / relative
        if cycle in SEALED_CYCLES:
            if not absolute.is_file():
                raise EvidenceError(f"sealed legacy cycle missing: V{cycle}")
            seal_bytes = _read_root_relative_regular_bytes(
                root, relative.as_posix(), f"V{cycle} legacy seal"
            )
            seal_payload = _strict_json_object(seal_bytes, f"V{cycle} legacy seal")
            _validate_legacy_seal(seal_payload, cycle)
            run_binding = _validate_legacy_run(
                root,
                cycle,
                expected_result_file_sha256=seal_payload["result_file_sha256"],
                expected_ledger_sha256=seal_payload["ledger_sha256"],
                expected_embedded_result_sha256=seal_payload["embedded_result_sha256"],
                expected_signal_id_set_sha256=seal_payload["signal_id_set_sha256"],
                expected_signals=seal_payload["signals"],
                expected_effective_bet_days=seal_payload["effective_bet_days"],
            )
            state = "LEGACY_SEALED_ORACLE_INPUTS_MISSING"
        elif cycle in FAILED_CYCLES:
            if _path_entry_exists(absolute):
                raise EvidenceError(f"failed legacy cycle unexpectedly has a seal: V{cycle}")
            unexpected_outputs = _failed_cycle_output_paths(root, cycle)
            if unexpected_outputs:
                raise EvidenceError(
                    f"V{cycle} failure receipt contradicts result/ledger files: "
                    f"{unexpected_outputs}"
                )
            receipt_relative = FAILED_CYCLE_RECEIPTS[cycle]
            receipt_bytes = _read_root_relative_regular_bytes(
                root, receipt_relative, f"V{cycle} failure receipt"
            )
            _validate_failed_cycle_receipt(
                _strict_json_object(receipt_bytes, f"V{cycle} failure receipt"),
                cycle,
            )
            failure_receipt_binding = {
                "path": receipt_relative,
                "sha256": sha256_bytes(receipt_bytes),
                "result_or_ledger_paths_found": [],
            }
            state = "LEGACY_TERMINAL_FAILURE_RECEIPT_BOUND_NO_RESULT"
        elif cycle in INVALID_CYCLES:
            if _path_entry_exists(absolute):
                raise EvidenceError(f"invalid legacy cycle unexpectedly has a seal: V{cycle}")
            state = "LEGACY_INVALID_NOT_ADMISSIBLE"
        else:
            raise EvidenceError(f"unclassified legacy cycle: V{cycle}")
        rows.append({
            "cycle": f"V{cycle}",
            "legacy_seal_path": relative.as_posix() if absolute.is_file() else None,
            "legacy_seal_sha256": (
                sha256_bytes(seal_bytes) if cycle in SEALED_CYCLES else None
            ),
            "legacy_run_binding": run_binding,
            "failure_receipt_binding": failure_receipt_binding,
            "coverage_state": state,
            "oracle_input_coverage": "MISSING",
            "missing_independent_oracle_inputs": missing_inputs,
            "official_oracle_pass": False,
            "retroactive_promotion_allowed": False,
        })
    v34_failure = root / "V34_RESULT_VALIDATION_FAILURE.json"
    if not v34_failure.is_file():
        raise EvidenceError("V34 validation-failure evidence is missing")
    v34_bytes = _read_root_relative_regular_bytes(
        root,
        "V34_RESULT_VALIDATION_FAILURE.json",
        "V34 validation-failure evidence",
    )
    v34_payload = _strict_json_object(v34_bytes, "V34 validation-failure evidence")
    _validate_v34_failure(v34_payload)
    v34_run_binding = _validate_legacy_run(
        root,
        34,
        expected_result_file_sha256=v34_payload["result_file_sha256"],
        expected_ledger_sha256=v34_payload["ledger_file_sha256"],
        expected_embedded_result_sha256=None,
        expected_signal_id_set_sha256=None,
        expected_signals=None,
        expected_effective_bet_days=None,
    )
    for row in rows:
        if row["cycle"] == "V34":
            row["legacy_run_binding"] = v34_run_binding
            break
    return seal({
        "schema_version": 2,
        "classification": "LEGACY_ORACLE_COVERAGE_SIDECAR_ONLY",
        "cycles": rows,
        "sealed_cycle_count": sum(
            row["coverage_state"] == "LEGACY_SEALED_ORACLE_INPUTS_MISSING"
            for row in rows
        ),
        "invalid_cycle_count": len(INVALID_CYCLES),
        "execution_failure_cycle_count": len(FAILED_CYCLES),
        "reconstructable_count": 0,
        "official_oracle_pass_count": 0,
        "frozen_artifact_count": len(frozen_artifact_hashes),
        "frozen_artifact_stream_sha256": frozen_artifact_set_sha256,
        "v34_validation_failure_sha256": sha256_bytes(v34_bytes),
        "legacy_seals_changed": False,
    }, "coverage_sha256")


def _validate_legacy_v1_audit(
    audit: Mapping[str, Any], raw_files: Mapping[str, bytes]
) -> str:
    if type(audit) is not dict or set(audit) != set(LEGACY_V1_AUDIT_KEYS):
        raise EvidenceError("legacy V1 audit schema changed")
    _exact_int(audit.get("schema_version"), 1, "legacy V1 audit schema")
    if audit.get("checkpoint_id") != "PAPER_RESEARCH_JPY_ORACLE_V1" \
            or audit.get("classification") != "FUTURE_ONLY_INDEPENDENT_ECONOMIC_ORACLE" \
            or audit.get("anchor_status") != "LOCAL_REPRODUCIBLE" \
            or audit.get("holdout_state") != "UNOPENED":
        raise EvidenceError("legacy V1 audit identity/classification changed")
    for field, expected in (
        ("producer_metrics_used", False),
        ("same_signal_ids_all_arms", True),
        ("all_proposals_have_all_arm_dispositions", True),
        ("official_strategy_run_performed", False),
        ("profit_evidence_generated", False),
        ("remote_anchor_required_for_external_status", True),
        ("legacy_seals_changed", False),
    ):
        _exact_bool(audit.get(field), expected, f"legacy V1 audit {field}")
    for field in (
        "terminal_inventory_mtm_jpy_micros", "external_orders",
        "legacy_official_oracle_pass_count",
    ):
        _exact_int(audit.get(field), 0, f"legacy V1 audit {field}")
    authority = audit.get("authority")
    if type(authority) is not dict or set(authority) != set(LEGACY_V1_AUTHORITY):
        raise EvidenceError("legacy V1 authority schema changed")
    for field, expected in LEGACY_V1_AUTHORITY.items():
        if field == "external_orders":
            _exact_int(authority[field], expected, f"legacy V1 authority {field}")
        else:
            _exact_bool(authority[field], expected, f"legacy V1 authority {field}")
    evidence_hashes = audit.get("evidence_artifact_sha256")
    expected_names = set(LEGACY_V1_EVIDENCE_NAMES) - {"oracle_checkpoint_v1.json"}
    expected_paths = {
        f"evidence/paper_research_jpy_oracle_v1/{name}" for name in expected_names
    }
    if type(evidence_hashes) is not dict or set(evidence_hashes) != expected_paths:
        raise EvidenceError("legacy V1 audit evidence map changed")
    for name in expected_names:
        relative = f"evidence/paper_research_jpy_oracle_v1/{name}"
        if evidence_hashes[relative] != sha256_bytes(raw_files[name]):
            raise EvidenceError(f"legacy V1 audit evidence hash mismatch: {name}")
    source_hashes_value = audit.get("source_artifact_sha256")
    if type(source_hashes_value) is not dict or not source_hashes_value:
        raise EvidenceError("legacy V1 audit source map is invalid")
    for name, value in source_hashes_value.items():
        if type(name) is not str or not name:
            raise EvidenceError("legacy V1 audit source path is invalid")
        _require_sha256(value, f"legacy V1 audit source {name}")
    for field in (
        "oracle_root_sha256", "verifier_receipt_sha256", "legacy_coverage_sha256",
    ):
        _require_sha256(audit.get(field), f"legacy V1 audit {field}")
    audit_sha256 = _require_sha256(
        audit.get("audit_sha256"), "legacy V1 embedded audit"
    )
    if audit_sha256 != embedded(audit, "audit_sha256") \
            or audit_sha256 != SUPERSEDED_V1_AUDIT_SHA256:
        raise EvidenceError("legacy V1 audit differs from its fixed review hash")
    return audit_sha256


def _legacy_v1_evidence_binding(root: Path) -> dict[str, Any]:
    root_fd, root_identity = _open_root_anchor(root)
    try:
        opened = _open_existing_parent_at(
            root_fd,
            ("evidence", "paper_research_jpy_oracle_v1", "__sentinel__"),
        )
        if opened is None:
            raise EvidenceError("legacy V1 evidence root is missing")
        evidence_fd, _ = opened
        try:
            names = tuple(sorted(os.listdir(evidence_fd)))
            if names != LEGACY_V1_EVIDENCE_NAMES:
                raise EvidenceError("legacy V1 evidence file set changed")
            for name in names:
                info = os.stat(name, dir_fd=evidence_fd, follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) \
                        or info.st_mode & 0o111 or info.st_mode & 0o022:
                    raise EvidenceError(
                        f"legacy V1 evidence mode/type changed: {name}"
                    )
            raw_files = {
                name: _read_regular_at(
                    evidence_fd, name, maximum_bytes=MAX_ARTIFACT_BYTES
                )
                for name in names
            }
        finally:
            os.close(evidence_fd)
        _assert_root_identity(root, root_identity, root_fd)
    finally:
        os.close(root_fd)
    files = {
        name: {
            "sha256": sha256_bytes(data),
            "size_bytes": len(data),
        }
        for name, data in raw_files.items()
    }
    audit = _strict_json_object(
        raw_files["oracle_checkpoint_v1.json"], "legacy V1 audit"
    )
    if sha256_bytes(raw_files["oracle_checkpoint_v1.json"]) \
            != SUPERSEDED_V1_AUDIT_FILE_SHA256:
        raise EvidenceError("legacy V1 audit file hash differs from fixed review")
    aggregate_sha256 = sha256_bytes(canonical(files))
    if aggregate_sha256 != SUPERSEDED_V1_AGGREGATE_SHA256:
        raise EvidenceError("legacy V1 aggregate differs from its fixed review hash")
    audit_sha256 = _validate_legacy_v1_audit(audit, raw_files)
    git_binding = _verify_legacy_v1_git_binding(root, raw_files)
    return {
        "root_relative_path": "evidence/paper_research_jpy_oracle_v1",
        "file_count": len(files),
        "files": files,
        "aggregate_sha256": aggregate_sha256,
        "prior_embedded_audit_sha256": audit_sha256,
        "audit_file_sha256": SUPERSEDED_V1_AUDIT_FILE_SHA256,
        "git_binding": git_binding,
    }


def _fixed_git_output(root: Path, arguments: tuple[str, ...], label: str) -> bytes:
    if not GIT.is_file():
        raise EvidenceError("fixed Git executable is unavailable")
    try:
        completed = subprocess.run(
            (str(GIT), "-C", str(root), *arguments),
            env={
                "PATH": "/usr/bin:/bin",
                "LANG": "C",
                "LC_ALL": "C",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_OPTIONAL_LOCKS": "0",
            },
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise EvidenceError(f"fixed Git {label} failed to execute") from error
    if completed.returncode != 0 or completed.stderr:
        raise EvidenceError(f"fixed Git {label} failed")
    return completed.stdout


def _fixed_git_oid(root: Path, revision: str, label: str) -> str:
    raw = _fixed_git_output(root, ("rev-parse", "--verify", revision), label)
    try:
        value = raw.decode("ascii").rstrip("\n")
    except UnicodeDecodeError as error:
        raise EvidenceError(f"fixed Git {label} returned non-ASCII") from error
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise EvidenceError(f"fixed Git {label} returned an invalid object ID")
    return value


def _verify_legacy_v1_git_binding(
    root: Path, raw_files: Mapping[str, bytes]
) -> dict[str, Any]:
    commit = _fixed_git_oid(root, f"{SUPERSEDED_COMMIT}^{{commit}}", "commit")
    tree = _fixed_git_oid(root, f"{SUPERSEDED_COMMIT}^{{tree}}", "commit tree")
    subtree = _fixed_git_oid(
        root,
        f"{SUPERSEDED_COMMIT}:{SUPERSEDED_V1_GIT_PATH}",
        "legacy V1 subtree",
    )
    if commit != SUPERSEDED_COMMIT \
            or tree != SUPERSEDED_COMMIT_TREE \
            or subtree != SUPERSEDED_V1_SUBTREE:
        raise EvidenceError("legacy V1 fixed Git commit/tree/subtree binding mismatch")
    tree_rows = _fixed_git_output(
        root,
        ("ls-tree", "--full-tree", "-z", SUPERSEDED_V1_SUBTREE),
        "legacy V1 tree entries",
    ).split(b"\0")
    if tree_rows[-1] != b"":
        raise EvidenceError("legacy V1 Git tree lacks terminal NUL")
    parsed_tree: dict[str, str] = {}
    for raw in tree_rows[:-1]:
        try:
            header, raw_name = raw.split(b"\t", 1)
            mode, object_type, oid = header.decode("ascii").split(" ")
            name = raw_name.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as error:
            raise EvidenceError("legacy V1 Git tree entry is malformed") from error
        if mode != "100644" or object_type != "blob" or name in parsed_tree:
            raise EvidenceError("legacy V1 Git tree mode/type/name changed")
        parsed_tree[name] = oid
    if set(parsed_tree) != set(LEGACY_V1_EVIDENCE_NAMES):
        raise EvidenceError("legacy V1 Git tree file set changed")
    blob_oids: dict[str, str] = {}
    for name in LEGACY_V1_EVIDENCE_NAMES:
        revision = f"{SUPERSEDED_COMMIT}:{SUPERSEDED_V1_GIT_PATH}/{name}"
        oid = _fixed_git_oid(root, revision, f"legacy V1 blob {name}")
        committed_bytes = _fixed_git_output(
            root, ("cat-file", "blob", revision), f"legacy V1 blob bytes {name}"
        )
        if committed_bytes != raw_files[name]:
            raise EvidenceError(f"legacy V1 working bytes differ from fixed Git blob: {name}")
        if parsed_tree[name] != oid:
            raise EvidenceError(f"legacy V1 tree/blob object mismatch: {name}")
        blob_oids[name] = oid
    return {
        "object_database_verified": True,
        "commit": commit,
        "commit_tree": tree,
        "subtree_path": SUPERSEDED_V1_GIT_PATH,
        "subtree": subtree,
        "file_mode": "100644",
        "blob_oids": dict(sorted(blob_oids.items())),
    }


def supersession(root: Path) -> dict[str, Any]:
    return seal({
        "schema_version": 2,
        "superseded_commit": SUPERSEDED_COMMIT,
        "classification": "SUPERSEDED_NOT_ADMISSIBLE",
        "reason_codes": [
            "COLLISION_POLICY_NOT_ENFORCED",
            "UNSIGNED_CURRENCY_EXPOSURE",
            "SOURCE_CLOCK_USED_FOR_AGE_AND_FINANCING",
            "INCOMPLETE_MONTH_GRID",
            "REALIZED_ONLY_DRAWDOWN",
            "MARGIN_RUIN_NOT_ENFORCED",
            "ADVERSE_ORDERING_NOT_ENFORCED",
            "PROPOSAL_PROVENANCE_UNBOUND",
            "TICK_PIP_SCALE_UNFROZEN",
            "UNTRUSTED_ROOT_OVERRIDE",
            "FILE_PUBLICATION_BOUNDARY_UNSAFE",
        ],
        "retroactive_promotion_allowed": False,
        "strategy_or_profit_evidence": False,
        "legacy_seals_changed": False,
        "legacy_v1_evidence_binding": _legacy_v1_evidence_binding(root),
    }, "supersession_sha256")


def source_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        try:
            value = _read_root_relative_regular_bytes(
                root,
                relative,
                "checkpoint source file",
                maximum_bytes=MAX_CODE_BYTES,
            )
        except EvidenceError as error:
            raise EvidenceError(
                f"checkpoint source file cannot be frozen: {relative}"
            ) from error
        result[relative] = sha256_bytes(value)
    return dict(sorted(result.items()))


def _assert_frozen_bindings(
    root: Path,
    expected_source_hashes: Mapping[str, str],
    expected_coverage_sha256: str,
    expected_supersession_sha256: str,
    phase: str,
) -> None:
    if source_hashes(root) != expected_source_hashes:
        raise EvidenceError(f"checkpoint source files changed {phase}")
    if legacy_coverage(root).get("coverage_sha256") != expected_coverage_sha256:
        raise EvidenceError(f"legacy cycle coverage changed {phase}")
    if supersession(root).get("supersession_sha256") != expected_supersession_sha256:
        raise EvidenceError(f"legacy V1 supersession evidence changed {phase}")


def _runtime_artifact_limit(relative: str) -> int:
    if relative.endswith("source_blob.jsonl") \
            or relative.endswith("oracle_ledger.jsonl"):
        return MAX_ARTIFACT_BYTES
    if relative.endswith(("oracle_code_snapshot.py", "reference_code_snapshot.py")):
        return MAX_CODE_BYTES
    return MAX_JSON_BYTES


def _assert_exact_private_tree(
    root: Path,
    expected_files: frozenset[str],
    label: str,
    seen_file_inodes: set[tuple[int, int]],
) -> None:
    _assert_private_directory(root, label)
    expected_directories = {
        "/".join(parts[:end])
        for relative in expected_files
        for parts in (_safe_relative(relative),)
        for end in range(1, len(parts))
    }
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    root_fd = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        _safe_directory_fd(root_fd, label)

        def walk(directory_fd: int, prefix: tuple[str, ...]) -> None:
            _safe_directory_fd(directory_fd, f"{label} directory fence")
            for name in sorted(os.listdir(directory_fd)):
                if type(name) is not str or name in {"", ".", ".."} \
                        or "/" in name:
                    raise EvidenceError(f"{label} contains an unsafe entry name")
                relative = "/".join((*prefix, name))
                named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISDIR(named.st_mode):
                    if named.st_uid != os.geteuid() \
                            or stat.S_IMODE(named.st_mode) != 0o700:
                        raise EvidenceError(
                            f"{label} contains an unsafe directory: {relative}"
                        )
                    child_fd = os.open(
                        name,
                        os.O_RDONLY
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=directory_fd,
                    )
                    try:
                        opened = _safe_directory_fd(
                            child_fd, f"{label} directory {relative}"
                        )
                        if stat.S_IMODE(opened.st_mode) != 0o700 \
                                or _directory_identity(opened) \
                                != _directory_identity(named):
                            raise EvidenceError(
                                f"{label} directory changed while opening: {relative}"
                            )
                        actual_directories.add(relative)
                        walk(child_fd, (*prefix, name))
                        after = _safe_directory_fd(
                            child_fd, f"{label} directory final fence {relative}"
                        )
                        named_after = os.stat(
                            name, dir_fd=directory_fd, follow_symlinks=False
                        )
                        if _directory_identity(after) \
                                != _directory_identity(opened) \
                                or _directory_identity(named_after) \
                                != _directory_identity(opened):
                            raise EvidenceError(
                                f"{label} directory changed during traversal: {relative}"
                            )
                    finally:
                        os.close(child_fd)
                    continue
                if not stat.S_ISREG(named.st_mode) \
                        or named.st_uid != os.geteuid() or named.st_nlink != 1 \
                        or stat.S_IMODE(named.st_mode) != 0o600:
                    raise EvidenceError(
                        f"{label} contains an unsafe file: {relative}"
                    )
                identity = (named.st_dev, named.st_ino)
                if identity in seen_file_inodes:
                    raise EvidenceError(
                        f"runtime scratch contains a hard-linked file: {relative}"
                    )
                seen_file_inodes.add(identity)
                _read_regular_at(
                    directory_fd,
                    name,
                    maximum_bytes=_runtime_artifact_limit(relative),
                )
                actual_files.add(relative)
            _safe_directory_fd(
                directory_fd, f"{label} directory traversal final fence"
            )

        walk(root_fd, ())
    finally:
        os.close(root_fd)
    _assert_private_directory(root, f"{label} final pathname fence")
    if actual_directories != expected_directories:
        raise EvidenceError(f"{label} directory set changed")
    if actual_files != set(expected_files):
        raise EvidenceError(f"{label} file set changed")


def _assert_exact_scratch_tree(
    scratch: Path,
    roots: Mapping[str, Path],
) -> None:
    _assert_private_directory(scratch, "runtime scratch root")
    expected_names = frozenset({
        "oracle_input_root",
        "oracle_output_root",
        "verifier_input_root",
        "verifier_output_root",
    })
    if frozenset(item.name for item in scratch.iterdir()) != expected_names:
        raise EvidenceError("runtime scratch top-level tree changed")
    if set(roots) != set(expected_names):
        raise EvidenceError("runtime root map changed")
    _assert_distinct_private_roots(roots)
    expected_by_root = {
        "oracle_input_root": ORACLE_INPUT_FILES,
        "oracle_output_root": ORACLE_OUTPUT_ROOT_FILES,
        "verifier_input_root": VERIFIER_INPUT_FILES,
        "verifier_output_root": VERIFIER_OUTPUT_ROOT_FILES,
    }
    seen_file_inodes: set[tuple[int, int]] = set()
    for name in sorted(expected_names):
        if roots[name].parent != scratch or roots[name].name != name:
            raise EvidenceError("runtime root path escaped the scratch root")
        _assert_exact_private_tree(
            roots[name],
            expected_by_root[name],
            name,
            seen_file_inodes,
        )


def _require_runtime_scratch_integrity(
    scratch: Path,
    roots: Mapping[str, Path],
) -> None:
    """Reopen and ACL-fence every producer-owned nested runtime directory."""
    nested_directories = (
        ("oracle_input_root", "inputs"),
        ("oracle_output_root", "oracle_output"),
        ("verifier_input_root", "inputs"),
        ("verifier_output_root", "verifier_output"),
    )
    if set(roots) != {
        "oracle_input_root",
        "oracle_output_root",
        "verifier_input_root",
        "verifier_output_root",
    }:
        raise EvidenceError("runtime root map changed")
    for root_name, nested_name in nested_directories:
        root_fd = os.open(
            roots[root_name],
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            root_before = _safe_directory_fd(
                root_fd, f"{root_name} nested-directory root fence"
            )
            if stat.S_IMODE(root_before.st_mode) != 0o700:
                raise EvidenceError(f"{root_name} is not owner-private")
            named_before = os.stat(
                nested_name, dir_fd=root_fd, follow_symlinks=False
            )
            if not stat.S_ISDIR(named_before.st_mode) \
                    or named_before.st_uid != os.geteuid() \
                    or stat.S_IMODE(named_before.st_mode) != 0o700:
                raise EvidenceError(
                    f"{root_name}/{nested_name} is not owner-private"
                )
            nested_fd = os.open(
                nested_name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_fd,
            )
            try:
                nested_before = _safe_directory_fd(
                    nested_fd, f"{root_name}/{nested_name} runtime fence"
                )
                if stat.S_IMODE(nested_before.st_mode) != 0o700 \
                        or _directory_identity(nested_before) \
                        != _directory_identity(named_before):
                    raise EvidenceError(
                        f"{root_name}/{nested_name} changed while opening"
                    )
                nested_after = _safe_directory_fd(
                    nested_fd, f"{root_name}/{nested_name} runtime final fence"
                )
                named_after = os.stat(
                    nested_name, dir_fd=root_fd, follow_symlinks=False
                )
                if _directory_identity(nested_after) \
                        != _directory_identity(nested_before) \
                        or _directory_identity(named_after) \
                        != _directory_identity(nested_before):
                    raise EvidenceError(
                        f"{root_name}/{nested_name} changed during runtime fence"
                    )
            finally:
                os.close(nested_fd)
            root_after = _safe_directory_fd(
                root_fd, f"{root_name} nested-directory root final fence"
            )
            if _directory_identity(root_after) != _directory_identity(root_before):
                raise EvidenceError(f"{root_name} changed during runtime fence")
        finally:
            os.close(root_fd)
    _assert_exact_scratch_tree(scratch, roots)


def _collect_runtime_artifacts(
    oracle_input_root: Path,
    oracle_output_root: Path,
    verifier_input_root: Path,
    verifier_output_root: Path,
) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for source_root, relative in (
        *((oracle_input_root, relative) for relative in (
        "inputs/source_blob.jsonl",
        "inputs/source_manifest.json",
        "inputs/instrument_registry.json",
        "inputs/proposal.json",
        "inputs/execution_policy.json",
        "inputs/inventory_policy.json",
        "inputs/accounting_policy.json",
        "inputs/evaluation_policy.json",
        "inputs/authority_policy.json",
        "inputs/oracle_request.json",
        )),
        *((verifier_input_root, relative) for relative in (
        "inputs/oracle_code_snapshot.py",
        "inputs/oracle_contract_snapshot.json",
        "inputs/oracle_schema_snapshot.json",
        "inputs/reference_code_snapshot.py",
        "inputs/reference_contract_snapshot.json",
        "inputs/verifier_request.json",
        )),
        *((oracle_output_root, relative) for relative in (
        "oracle_output/intent.json",
        "oracle_output/oracle_ledger.jsonl",
        "oracle_output/oracle_manifest.json",
        "oracle_output/COMMIT.json",
        )),
        *((verifier_output_root, relative) for relative in (
        "verifier_output/verifier_receipt.json",
        "verifier_output/COMMIT.json",
        )),
    ):
        path = source_root / relative
        result[f"{EVIDENCE_ROOT}/{relative}"] = read_regular_bytes(
            path,
            f"sealed runtime artifact {relative}",
            maximum_bytes=_runtime_artifact_limit(relative),
        )
    expected = {
        f"{EVIDENCE_ROOT}/{relative}" for relative in RUNTIME_ARTIFACT_FILES
    }
    if set(result) != expected or len(result) != EXPECTED_RUNTIME_ARTIFACT_COUNT:
        raise EvidenceError("sealed runtime artifact set/count changed")
    return result


def _read_exact_output_set(
    output_root: Path, directory: str, expected: frozenset[str], label: str
) -> dict[str, bytes]:
    path = output_root / directory
    _assert_private_directory(path, f"{label} directory")
    names = frozenset(item.name for item in path.iterdir())
    if names != expected:
        raise EvidenceError(f"{label} file set changed")
    return {
        name: read_regular_bytes(
            path / name,
            f"{label} {name}",
            maximum_bytes=(
                MAX_ARTIFACT_BYTES if name == "oracle_ledger.jsonl" else MAX_JSON_BYTES
            ),
        )
        for name in sorted(expected)
    }


def _validate_described_artifacts(
    state_root: Path,
    request: Mapping[str, Any],
    labels: tuple[str, ...],
    label: str,
) -> dict[str, str]:
    hashes = _artifact_descriptor_hashes(request, labels, label)
    seen_paths: set[str] = set()
    for name in labels:
        descriptor = request[name]
        if descriptor["relative_path"] in seen_paths:
            raise EvidenceError(f"{label} aliases multiple artifacts to one path")
        seen_paths.add(descriptor["relative_path"])
        parts = _safe_relative(descriptor["relative_path"])
        path = state_root.joinpath(*parts)
        data = read_regular_bytes(
            path,
            f"{label} {name}",
            maximum_bytes=(
                MAX_ARTIFACT_BYTES
                if name in {"source_blob", "oracle_ledger"}
                else MAX_CODE_BYTES
                if name in {"oracle_code_snapshot", "reference_code_snapshot"}
                else MAX_JSON_BYTES
            ),
        )
        if sha256_bytes(data) != descriptor["sha256"] \
                or len(data) != descriptor["size_bytes"]:
            raise EvidenceError(f"{label} {name} bytes differ from descriptor")
    return hashes


def _validate_ledger_chain(ledger: bytes) -> tuple[int, str]:
    if not ledger or not ledger.endswith(b"\n"):
        raise EvidenceError("Oracle ledger must be nonempty canonical JSONL")
    previous = "0" * 64
    count = 0
    for count, line in enumerate(ledger.splitlines(), start=1):
        row = _canonical_json_object(line + b"\n", f"Oracle ledger row {count}")
        _require_sha256(row.get("previous_hash"), f"Oracle ledger row {count} previous")
        record_hash = _require_sha256(
            row.get("record_hash"), f"Oracle ledger row {count} record"
        )
        if row["previous_hash"] != previous \
                or record_hash != embedded(row, "record_hash"):
            raise EvidenceError(f"Oracle ledger hash chain failed at row {count}")
        _exact_int(row.get("ledger_sequence"), count, f"Oracle ledger row {count} sequence")
        _exact_int(
            row.get("external_order_count"), 0,
            f"Oracle ledger row {count} external orders",
        )
        previous = record_hash
    return count, previous


def _validate_runtime_metrics(metrics: Any) -> None:
    if type(metrics) is not dict or set(metrics) != set(ORACLE_METRICS_KEYS):
        raise EvidenceError("Oracle metrics schema changed")
    _exact_int(metrics.get("schema_version"), 2, "Oracle metrics schema")
    initial_equity = metrics.get("initial_equity_jpy_micros")
    if type(initial_equity) is not int or initial_equity <= 0:
        raise EvidenceError("Oracle metrics initial equity must be a positive integer")
    if metrics.get("metrics_sha256") != embedded(metrics, "metrics_sha256"):
        raise EvidenceError("Oracle metrics self-hash mismatch")
    for field in (
        "same_signal_ids_all_arms",
        "all_proposals_have_all_arm_dispositions",
        "common_gross_reference_shared",
    ):
        _exact_bool(metrics.get(field), True, f"Oracle metrics {field}")
    _exact_int(metrics.get("external_orders"), 0, "Oracle metrics external orders")
    _exact_int(
        metrics.get("terminal_inventory_mtm_jpy_micros"),
        0,
        "Oracle metrics terminal inventory MTM",
    )
    arms = metrics.get("arms")
    if type(arms) is not dict or set(arms) != {
        "RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS",
    }:
        raise EvidenceError("Oracle metrics arm set changed")
    signal_hashes: set[str] = set()
    for arm, arm_metrics in arms.items():
        if type(arm_metrics) is not dict \
                or set(arm_metrics) != set(ORACLE_ARM_METRICS_KEYS):
            raise EvidenceError(f"Oracle {arm} metrics schema changed")
        integer_fields = (
            "proposal_count", "executed_count",
            "common_gross_pnl_jpy_micros", "realized_cost_jpy_micros",
            "fill_sizing_drag_jpy_micros",
            "latency_spread_slippage_drag_jpy_micros",
            "direct_commission_financing_cost_jpy_micros",
            "admission_opportunity_drag_jpy_micros",
            "total_execution_and_admission_drag_jpy_micros",
            "net_pnl_jpy_micros", "ending_equity_jpy_micros",
            "max_drawdown_jpy_micros", "cvar_tail_bps",
            "cluster_cvar_jpy_micros", "currency_time_cluster_n_eff",
            "max_gross_notional_jpy_micros",
            "minimum_marked_equity_jpy_micros",
            "maximum_required_margin_jpy_micros",
            "minimum_free_margin_jpy_micros", "terminal_open_positions",
            "terminal_inventory_mtm_jpy_micros",
        )
        for field in integer_fields:
            if type(arm_metrics.get(field)) is not int:
                raise EvidenceError(f"Oracle {arm} {field} must be an integer")
        if arm_metrics["proposal_count"] < 0 \
                or arm_metrics["executed_count"] < 0 \
                or arm_metrics["executed_count"] > arm_metrics["proposal_count"] \
                or arm_metrics["currency_time_cluster_n_eff"] < 0 \
                or arm_metrics["terminal_open_positions"] < 0:
            raise EvidenceError(f"Oracle {arm} count metrics are invalid")
        for field in (
            "ending_equity_multiple", "direction_accuracy",
            "max_drawdown_ratio", "cluster_cvar_return",
        ):
            _fixed_decimal_text(arm_metrics.get(field), f"Oracle {arm} {field}")
        disposition_counts = arm_metrics.get("disposition_counts")
        if type(disposition_counts) is not dict or not disposition_counts:
            raise EvidenceError(f"Oracle {arm} dispositions are invalid")
        for disposition, count in disposition_counts.items():
            if type(disposition) is not str or not disposition \
                    or type(count) is not int or count < 0:
                raise EvidenceError(f"Oracle {arm} disposition count is invalid")
        if sum(disposition_counts.values()) != arm_metrics["proposal_count"]:
            raise EvidenceError(f"Oracle {arm} disposition total is inconsistent")
        observations = arm_metrics.get("currency_time_cluster_observations")
        if type(observations) is not list:
            raise EvidenceError(f"Oracle {arm} cluster observations are invalid")
        cluster_ids: set[str] = set()
        for index, observation in enumerate(observations, start=1):
            if type(observation) is not dict \
                    or set(observation) != set(ORACLE_CLUSTER_OBSERVATION_KEYS):
                raise EvidenceError(
                    f"Oracle {arm} cluster observation {index} schema changed"
                )
            cluster_id = _require_sha256(
                observation.get("cluster_id"),
                f"Oracle {arm} cluster observation {index} ID",
            )
            if cluster_id in cluster_ids:
                raise EvidenceError(f"Oracle {arm} cluster IDs are duplicated")
            cluster_ids.add(cluster_id)
            _require_sha256(
                observation.get("source_signal_set_sha256"),
                f"Oracle {arm} cluster observation {index} signal set",
            )
            if type(observation.get("time_bucket")) is not int:
                raise EvidenceError(
                    f"Oracle {arm} cluster observation {index} time bucket is invalid"
                )
            currency_nodes = observation.get("currency_nodes")
            if type(currency_nodes) is not list or not currency_nodes \
                    or any(type(node) is not str or not node for node in currency_nodes) \
                    or currency_nodes != sorted(set(currency_nodes)):
                raise EvidenceError(
                    f"Oracle {arm} cluster observation {index} currencies are invalid"
                )
            for field in (
                "ledger_net_pnl_jpy_micros",
                "cluster_risk_net_pnl_jpy_micros",
            ):
                if type(observation.get(field)) is not int:
                    raise EvidenceError(
                        f"Oracle {arm} cluster observation {index} {field} is invalid"
                    )
            _fixed_decimal_text(
                observation.get("signed_return"),
                f"Oracle {arm} cluster observation {index} signed return",
            )
        if arm_metrics["currency_time_cluster_n_eff"] > len(observations):
            raise EvidenceError(f"Oracle {arm} cluster N_eff exceeds observations")
        monthly = arm_metrics.get("monthly")
        if type(monthly) is not list or not monthly:
            raise EvidenceError(f"Oracle {arm} monthly metrics are invalid")
        month_ids: list[str] = []
        for index, month in enumerate(monthly, start=1):
            if type(month) is not dict \
                    or set(month) != set(ORACLE_MONTHLY_METRICS_KEYS):
                raise EvidenceError(f"Oracle {arm} month {index} schema changed")
            month_id = month.get("month_id")
            if type(month_id) is not str or len(month_id) != 7 \
                    or month_id[4:5] != "-" or not month_id[:4].isdigit() \
                    or not month_id[5:].isdigit() \
                    or not 1 <= int(month_id[5:]) <= 12:
                raise EvidenceError(f"Oracle {arm} month {index} identity is invalid")
            month_ids.append(month_id)
            _exact_bool(
                month.get("comparable_full_month"), True,
                f"Oracle {arm} month {index} comparability",
            )
            _exact_bool(
                month.get("ruin_observed"), False,
                f"Oracle {arm} month {index} ruin state",
            )
            for field in (
                "segment_start_ts_ns", "segment_end_ts_ns",
                "start_equity_jpy_micros", "end_equity_jpy_micros",
            ):
                if type(month.get(field)) is not int:
                    raise EvidenceError(
                        f"Oracle {arm} month {index} {field} must be an integer"
                    )
            if month["segment_start_ts_ns"] >= month["segment_end_ts_ns"]:
                raise EvidenceError(f"Oracle {arm} month {index} interval is invalid")
            _fixed_decimal_text(
                month.get("equity_multiple"),
                f"Oracle {arm} month {index} equity multiple",
            )
            if month.get("equity_multiple_status") != "DEFINED":
                raise EvidenceError(f"Oracle {arm} month {index} is not defined")
        if month_ids != sorted(set(month_ids)):
            raise EvidenceError(f"Oracle {arm} month ordering/identity changed")
        _exact_int(
            arm_metrics.get("terminal_inventory_mtm_jpy_micros"), 0,
            f"Oracle {arm} terminal inventory MTM",
        )
        _exact_int(
            arm_metrics.get("terminal_open_positions"), 0,
            f"Oracle {arm} terminal positions",
        )
        _exact_bool(
            arm_metrics.get("margin_guard_pass"), True,
            f"Oracle {arm} margin guard",
        )
        signal_hashes.add(_require_sha256(
            arm_metrics.get("signal_id_set_sha256"),
            f"Oracle {arm} signal-ID set",
        ))
        if arm_metrics["ending_equity_jpy_micros"] \
                != initial_equity + arm_metrics["net_pnl_jpy_micros"]:
            raise EvidenceError(f"Oracle {arm} ending equity reconciliation failed")
        if arm_metrics["realized_cost_jpy_micros"] \
                != arm_metrics["fill_sizing_drag_jpy_micros"] \
                + arm_metrics["latency_spread_slippage_drag_jpy_micros"] \
                + arm_metrics["direct_commission_financing_cost_jpy_micros"] \
                + arm_metrics["admission_opportunity_drag_jpy_micros"] \
                or arm_metrics["total_execution_and_admission_drag_jpy_micros"] \
                != arm_metrics["realized_cost_jpy_micros"] \
                or arm_metrics["net_pnl_jpy_micros"] \
                != arm_metrics["common_gross_pnl_jpy_micros"] \
                - arm_metrics["realized_cost_jpy_micros"]:
            raise EvidenceError(f"Oracle {arm} PnL/cost reconciliation failed")
    if len(signal_hashes) != 1:
        raise EvidenceError("Oracle arm signal-ID sets diverged")
    raw = arms["RAW_SIGNAL"]
    base = arms["EXECUTABLE_BASE"]
    adverse = arms["ADVERSE_STRESS"]
    for field in (
        "realized_cost_jpy_micros",
        "fill_sizing_drag_jpy_micros",
        "latency_spread_slippage_drag_jpy_micros",
        "direct_commission_financing_cost_jpy_micros",
        "admission_opportunity_drag_jpy_micros",
        "total_execution_and_admission_drag_jpy_micros",
    ):
        values = (raw.get(field), base.get(field), adverse.get(field))
        if any(type(value) is not int for value in values) \
                or not (values[0] <= values[1] <= values[2]):
            raise EvidenceError(f"Oracle cost-arm ordering failed for {field}")
    if not (
        raw["common_gross_pnl_jpy_micros"]
        == base["common_gross_pnl_jpy_micros"]
        == adverse["common_gross_pnl_jpy_micros"]
    ) or len({
        raw["proposal_count"], base["proposal_count"], adverse["proposal_count"]
    }) != 1:
        raise EvidenceError("Oracle arms do not share one gross/proposal stream")
    if base["realized_cost_jpy_micros"] <= raw["realized_cost_jpy_micros"] \
            or adverse["realized_cost_jpy_micros"] \
                <= base["realized_cost_jpy_micros"]:
        raise EvidenceError("Oracle BASE/ADVERSE cost stress is not strictly ordered")


def _reference_result_snapshot_sha256(
    snapshot: Mapping[str, Any], ledger: bytes
) -> str:
    if type(snapshot) is not dict \
            or set(snapshot) != set(REFERENCE_RESULT_SNAPSHOT_KEYS):
        raise EvidenceError("reference result snapshot schema changed")
    if type(ledger) is not bytes \
            or type(snapshot.get("ledger_bytes_base64")) is not str:
        raise EvidenceError("reference result ledger snapshot type changed")
    encoded = snapshot["ledger_bytes_base64"]
    try:
        encoded_bytes = encoded.encode("ascii")
        decoded = base64.b64decode(encoded_bytes, validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        raise EvidenceError("reference result ledger base64 is invalid") from error
    if decoded != ledger or base64.b64encode(decoded) != encoded_bytes:
        raise EvidenceError(
            "reference result ledger base64 is not canonical standard padded ASCII"
        )
    snapshot_bytes = canonical(snapshot) + b"\n"
    if _canonical_json_object(
        snapshot_bytes,
        "reference result snapshot",
        REFERENCE_RESULT_SNAPSHOT_KEYS,
    ) != snapshot:
        raise EvidenceError("reference result snapshot canonical round-trip changed")
    return sha256_bytes(snapshot_bytes)


def _validate_inner_runtime_outputs(
    oracle_input_root: Path,
    oracle_output_root: Path,
    verifier_input_root: Path,
    verifier_output_root: Path,
    frozen_source_hashes: Mapping[str, str],
    oracle_launch: Mapping[str, Any],
    verifier_launch: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    oracle_files = _read_exact_output_set(
        oracle_output_root, "oracle_output", ORACLE_OUTPUT_FILES, "Oracle output"
    )
    oracle_request_bytes = read_regular_bytes(
        oracle_input_root / "inputs/oracle_request.json",
        "Oracle request",
        maximum_bytes=MAX_JSON_BYTES,
    )
    oracle_request = _canonical_json_object(
        oracle_request_bytes,
        "Oracle request",
        {"schema_version", *ORACLE_INPUT_LABELS, "output_directory"},
    )
    _exact_int(oracle_request.get("schema_version"), 2, "Oracle request schema")
    if oracle_request.get("output_directory") != "oracle_output":
        raise EvidenceError("Oracle request output directory changed")
    oracle_input_hashes = _validate_described_artifacts(
        oracle_input_root, oracle_request, ORACLE_INPUT_LABELS, "Oracle request"
    )
    authority_parts = _safe_relative(
        oracle_request["authority_policy"]["relative_path"]
    )
    authority_policy = _canonical_json_object(
        read_regular_bytes(
            oracle_input_root.joinpath(*authority_parts),
            "Oracle authority policy",
            maximum_bytes=MAX_JSON_BYTES,
        ),
        "Oracle authority policy",
        AUTHORITY_POLICY_KEYS,
    )
    _exact_int(
        authority_policy.get("schema_version"), 2,
        "Oracle authority policy schema",
    )
    if authority_policy.get("policy_id") != "FROZEN_PAPER_AUTHORITY_V1" \
            or authority_policy.get("authority_policy_sha256") \
                != embedded(authority_policy, "authority_policy_sha256"):
        raise EvidenceError("Oracle authority policy identity/self-hash mismatch")
    _validate_exact_authority(
        {field: authority_policy[field] for field in PAPER_ONLY_AUTHORITY},
        "Oracle authority policy",
    )
    request_sha256 = sha256_bytes(oracle_request_bytes)

    intent = _canonical_json_object(
        oracle_files["intent.json"], "Oracle intent", ORACLE_INTENT_KEYS
    )
    _exact_int(intent.get("schema_version"), 1, "Oracle intent schema")
    expected_intent_core = {
        "request_sha256": request_sha256,
        "code_sha256": frozen_source_hashes[ORACLE_PATH],
        "contract_sha256": frozen_source_hashes[CONTRACT_PATH],
        "schema_sha256": frozen_source_hashes[SCHEMA_PATH],
    }
    expected_transaction = sha256_bytes(canonical(expected_intent_core))
    if any(intent.get(field) != value for field, value in expected_intent_core.items()) \
            or intent.get("transaction_id") != expected_transaction:
        raise EvidenceError("Oracle intent is not bound to the frozen request/release")

    ledger = oracle_files["oracle_ledger.jsonl"]
    ledger_count, ledger_terminal = _validate_ledger_chain(ledger)
    manifest = _canonical_json_object(
        oracle_files["oracle_manifest.json"],
        "Oracle manifest",
        ORACLE_MANIFEST_KEYS,
    )
    _exact_int(manifest.get("schema_version"), 2, "Oracle manifest schema")
    if manifest.get("oracle_implementation") != "INDEPENDENT_JPY_ORACLE_V2" \
            or manifest.get("status") != "COMPLETE" \
            or manifest.get("classification") != RUNTIME_CLASSIFICATION \
            or manifest.get("anchor_status") != RUNTIME_ANCHOR_STATUS \
            or manifest.get("oracle_ledger_file") != "oracle_ledger.jsonl":
        raise EvidenceError("Oracle identity/classification boundary mismatch")
    for field, expected in (
        ("causal_signal_admission", False),
        ("release_evidence_eligible", False),
        ("detector_replay_receipt_required", True),
        ("producer_result_or_metrics_used", False),
        ("proposal_identity_generated_by_oracle", True),
    ):
        _exact_bool(manifest.get(field), expected, f"Oracle manifest {field}")
    _exact_int(manifest.get("external_orders"), 0, "Oracle manifest external orders")
    _exact_int(
        manifest.get("terminal_inventory_mtm_jpy_micros"), 0,
        "Oracle manifest terminal inventory MTM",
    )
    _exact_int(
        manifest.get("oracle_ledger_size_bytes"), len(ledger),
        "Oracle manifest ledger size",
    )
    _exact_int(
        manifest.get("oracle_ledger_row_count"), ledger_count,
        "Oracle manifest ledger row count",
    )
    _validate_exact_authority(manifest.get("authority"), "Oracle manifest")
    if manifest.get("oracle_root_sha256") != embedded(manifest, "oracle_root_sha256"):
        raise EvidenceError("Oracle manifest self-hash mismatch")
    if manifest.get("request_sha256") != request_sha256 \
            or manifest.get("input_artifact_sha256") != oracle_input_hashes \
            or manifest.get("raw_source_manifest_sha256") \
                != oracle_input_hashes["source_manifest"] \
            or manifest.get("oracle_ledger_sha256") != sha256_bytes(ledger) \
            or manifest.get("oracle_ledger_size_bytes") != len(ledger) \
            or manifest.get("oracle_ledger_row_count") != ledger_count \
            or manifest.get("oracle_ledger_terminal_hash") != ledger_terminal:
        raise EvidenceError("Oracle manifest request/ledger chain binding mismatch")
    _require_sha256(
        manifest.get("proposal_provenance_root_sha256"),
        "Oracle proposal provenance root",
    )
    _validate_runtime_metrics(manifest.get("oracle_metrics"))
    oracle_content_binding = manifest.get("oracle_release_content_binding")
    if type(oracle_content_binding) is not dict \
            or set(oracle_content_binding) \
                != set(ORACLE_RELEASE_CONTENT_BINDING_KEYS) \
            or oracle_content_binding != {
                "code_sha256": frozen_source_hashes[ORACLE_PATH],
                "contract_sha256": frozen_source_hashes[CONTRACT_PATH],
                "schema_sha256": frozen_source_hashes[SCHEMA_PATH],
                "launcher_sha256": oracle_launch["launcher_sha256"],
                "snapshot_mode": "SEALED_FD_COMPILE_EXEC_V2",
            }:
        raise EvidenceError("Oracle release-content binding mismatch")
    if manifest.get("oracle_execution_provenance_scope") \
            != EXECUTION_PROVENANCE_SCOPE:
        raise EvidenceError("Oracle overclaimed execution provenance")

    oracle_commit = _canonical_json_object(
        oracle_files["COMMIT.json"], "Oracle COMMIT", ORACLE_COMMIT_KEYS
    )
    _exact_int(oracle_commit.get("schema_version"), 1, "Oracle COMMIT schema")
    _exact_int(
        oracle_commit.get("ledger_size_bytes"), len(ledger),
        "Oracle COMMIT ledger size",
    )
    _exact_int(
        oracle_commit.get("manifest_size_bytes"),
        len(oracle_files["oracle_manifest.json"]),
        "Oracle COMMIT manifest size",
    )
    if oracle_commit != {
        "schema_version": 1,
        "transaction_id": expected_transaction,
        "request_sha256": request_sha256,
        "intent_sha256": sha256_bytes(oracle_files["intent.json"]),
        "ledger_sha256": sha256_bytes(ledger),
        "ledger_size_bytes": len(ledger),
        "manifest_sha256": sha256_bytes(oracle_files["oracle_manifest.json"]),
        "manifest_size_bytes": len(oracle_files["oracle_manifest.json"]),
        "terminal_hash": ledger_terminal,
    }:
        raise EvidenceError("Oracle COMMIT chain binding mismatch")

    verifier_files = _read_exact_output_set(
        verifier_output_root,
        "verifier_output",
        VERIFIER_OUTPUT_FILES,
        "verifier output",
    )
    verifier_request_bytes = read_regular_bytes(
        verifier_input_root / "inputs/verifier_request.json",
        "verifier request",
        maximum_bytes=MAX_JSON_BYTES,
    )
    verifier_request = _canonical_json_object(
        verifier_request_bytes,
        "verifier request",
        {"schema_version", *VERIFIER_INPUT_LABELS, "output_directory"},
    )
    _exact_int(verifier_request.get("schema_version"), 2, "verifier request schema")
    if verifier_request.get("output_directory") != "verifier_output":
        raise EvidenceError("verifier request output directory changed")
    verifier_input_hashes = _validate_described_artifacts(
        verifier_input_root,
        verifier_request,
        VERIFIER_INPUT_LABELS,
        "verifier request",
    )
    verifier_request_sha256 = sha256_bytes(verifier_request_bytes)
    for label, expected_path in (
        ("reference_code_snapshot", "inputs/reference_code_snapshot.py"),
        ("reference_contract_snapshot", "inputs/reference_contract_snapshot.json"),
    ):
        if verifier_request[label].get("relative_path") != expected_path:
            raise EvidenceError(f"verifier request {label} path changed")
    if verifier_input_hashes["oracle_request"] != request_sha256 \
            or verifier_input_hashes["oracle_code_snapshot"] \
                != frozen_source_hashes[ORACLE_PATH] \
            or verifier_input_hashes["oracle_contract_snapshot"] \
                != frozen_source_hashes[CONTRACT_PATH] \
            or verifier_input_hashes["oracle_schema_snapshot"] \
                != frozen_source_hashes[SCHEMA_PATH] \
            or verifier_input_hashes["reference_code_snapshot"] \
                != frozen_source_hashes[REFERENCE_PATH] \
            or verifier_input_hashes["reference_contract_snapshot"] \
                != frozen_source_hashes[REFERENCE_CONTRACT_PATH] \
            or verifier_input_hashes["oracle_intent"] \
                != sha256_bytes(oracle_files["intent.json"]) \
            or verifier_input_hashes["oracle_commit"] \
                != sha256_bytes(oracle_files["COMMIT.json"]) \
            or verifier_input_hashes["oracle_ledger"] != sha256_bytes(ledger) \
            or verifier_input_hashes["oracle_manifest"] \
                != sha256_bytes(oracle_files["oracle_manifest.json"]):
        raise EvidenceError("verifier request is not bound to the exact Oracle chain")

    receipt = _canonical_json_object(
        verifier_files["verifier_receipt.json"],
        "verifier receipt",
        VERIFIER_RECEIPT_KEYS,
    )
    _exact_int(receipt.get("schema_version"), 2, "verifier receipt schema")
    if receipt.get("verifier_implementation") \
            != "INDEPENDENT_JPY_ORACLE_VERIFIER_V2" \
            or receipt.get("status") != "VERIFIED_ACCOUNTING_ONLY" \
            or receipt.get("classification") != RUNTIME_CLASSIFICATION \
            or receipt.get("anchor_status") != RUNTIME_ANCHOR_STATUS:
        raise EvidenceError("verifier identity/classification boundary mismatch")
    for field, expected in (
        ("causal_signal_admission", False),
        ("release_evidence_eligible", False),
        ("admission_eligible", False),
        ("detector_replay_receipt_required", True),
        ("independently_rebuilt_ledger", True),
        ("independently_rebuilt_metrics", True),
        ("producer_result_or_metrics_used", False),
        ("reference_all_transactions_balanced", True),
        ("reference_accounting_diagnostics_only", True),
        ("reference_n_eff_statistical_admission_allowed", False),
        ("reference_direction_accuracy_profit_gate_allowed", False),
    ):
        _exact_bool(receipt.get(field), expected, f"verifier receipt {field}")
    _exact_int(receipt.get("external_orders"), 0, "verifier receipt external orders")
    _exact_int(
        receipt.get("terminal_inventory_mtm_jpy_micros"), 0,
        "verifier receipt terminal inventory MTM",
    )
    _exact_int(
        receipt.get("oracle_manifest_size_bytes"),
        len(oracle_files["oracle_manifest.json"]),
        "verifier receipt Oracle manifest size",
    )
    _exact_int(
        receipt.get("oracle_ledger_size_bytes"), len(ledger),
        "verifier receipt Oracle ledger size",
    )
    _exact_int(
        receipt.get("reference_journal_transaction_count"),
        REFERENCE_JOURNAL_TRANSACTION_COUNT,
        "verifier receipt reference journal transaction count",
    )
    _validate_exact_authority(receipt.get("authority"), "verifier receipt")
    if receipt.get("verifier_receipt_sha256") \
            != embedded(receipt, "verifier_receipt_sha256"):
        raise EvidenceError("verifier receipt self-hash mismatch")
    if receipt.get("input_artifact_sha256") != verifier_input_hashes \
            or receipt.get("oracle_request_sha256") != request_sha256 \
            or receipt.get("oracle_manifest_sha256") \
                != sha256_bytes(oracle_files["oracle_manifest.json"]) \
            or receipt.get("oracle_manifest_size_bytes") \
                != len(oracle_files["oracle_manifest.json"]) \
            or receipt.get("oracle_ledger_sha256") != sha256_bytes(ledger) \
            or receipt.get("expected_canonical_ledger_sha256") != sha256_bytes(ledger) \
            or receipt.get("oracle_ledger_size_bytes") != len(ledger) \
            or receipt.get("oracle_ledger_terminal_hash") != ledger_terminal \
            or receipt.get("oracle_root_sha256") != manifest["oracle_root_sha256"] \
            or receipt.get("raw_source_manifest_sha256") \
                != oracle_input_hashes["source_manifest"] \
            or receipt.get("verified_oracle_metrics") != manifest["oracle_metrics"] \
            or receipt.get("oracle_release_content_binding") \
                != oracle_content_binding \
            or receipt.get("oracle_execution_provenance_scope") \
                != EXECUTION_PROVENANCE_SCOPE:
        raise EvidenceError("verifier receipt is not bound to the exact Oracle chain")
    expected_reference_input_root = sha256_bytes(canonical({
        "artifact_sha256": dict(sorted(oracle_input_hashes.items())),
    }))
    reference_journal_root = _require_sha256(
        receipt.get("reference_journal_root_sha256"),
        "verifier receipt reference journal root",
    )
    reference_projection = {
        "all_transactions_balanced": True,
        "engine_id": REFERENCE_ENGINE_ID,
        "input_root_sha256": expected_reference_input_root,
        "journal_root_sha256": reference_journal_root,
        "journal_transaction_count": REFERENCE_JOURNAL_TRANSACTION_COUNT,
        "ledger_row_count": ledger_count,
        "ledger_sha256": sha256_bytes(ledger),
        "ledger_terminal_hash": ledger_terminal,
        "oracle_metrics_sha256": manifest["oracle_metrics"]["metrics_sha256"],
        "proposal_provenance_root_sha256": manifest[
            "proposal_provenance_root_sha256"
        ],
    }
    expected_reference_projection_sha256 = sha256_bytes(
        canonical(reference_projection)
    )
    ledger_bytes_base64 = base64.b64encode(ledger).decode("ascii")
    reference_result_snapshot = {
        "engine_id": REFERENCE_ENGINE_ID,
        "input_root_sha256": expected_reference_input_root,
        "ledger_bytes_base64": ledger_bytes_base64,
        "ledger_row_count": ledger_count,
        "ledger_terminal_hash": ledger_terminal,
        "oracle_metrics": manifest["oracle_metrics"],
        "proposal_provenance_root_sha256": manifest[
            "proposal_provenance_root_sha256"
        ],
        "journal_root_sha256": reference_journal_root,
        "journal_transaction_count": REFERENCE_JOURNAL_TRANSACTION_COUNT,
        "all_transactions_balanced": True,
        "economic_projection_sha256": expected_reference_projection_sha256,
    }
    expected_reference_result_sha256 = _reference_result_snapshot_sha256(
        reference_result_snapshot,
        ledger,
    )
    if receipt.get("reference_engine_id") != REFERENCE_ENGINE_ID \
            or receipt.get("reference_code_sha256") \
                != frozen_source_hashes[REFERENCE_PATH] \
            or receipt.get("reference_contract_sha256") \
                != frozen_source_hashes[REFERENCE_CONTRACT_PATH] \
            or receipt.get("reference_input_root_sha256") \
                != expected_reference_input_root \
            or receipt.get("reference_economic_projection_sha256") \
                != expected_reference_projection_sha256 \
            or receipt.get("reference_result_sha256") \
                != expected_reference_result_sha256:
        raise EvidenceError("verifier reference diagnostic chain binding mismatch")
    verifier_content_binding = receipt.get("verifier_release_content_binding")
    if type(verifier_content_binding) is not dict \
            or set(verifier_content_binding) \
                != set(VERIFIER_RELEASE_CONTENT_BINDING_KEYS) \
            or verifier_content_binding != {
                "code_sha256": frozen_source_hashes[VERIFIER_PATH],
                "schema_sha256": frozen_source_hashes[VERIFIER_SCHEMA_PATH],
                "launcher_sha256": verifier_launch["launcher_sha256"],
                "reference_code_sha256": frozen_source_hashes[REFERENCE_PATH],
                "reference_contract_sha256": frozen_source_hashes[
                    REFERENCE_CONTRACT_PATH
                ],
                "reference_result_sha256": expected_reference_result_sha256,
                "snapshot_mode": "SEALED_FD_COMPILE_EXEC_V2",
            }:
        raise EvidenceError("verifier release-content binding mismatch")
    if receipt.get("verifier_execution_provenance_scope") \
            != EXECUTION_PROVENANCE_SCOPE:
        raise EvidenceError("verifier overclaimed execution provenance")

    verifier_commit = _canonical_json_object(
        verifier_files["COMMIT.json"], "verifier COMMIT", VERIFIER_COMMIT_KEYS
    )
    _exact_int(verifier_commit.get("schema_version"), 2, "verifier COMMIT schema")
    _exact_int(
        verifier_commit.get("receipt_size_bytes"),
        len(verifier_files["verifier_receipt.json"]),
        "verifier COMMIT receipt size",
    )
    if verifier_commit != {
        "schema_version": 2,
        "request_sha256": verifier_request_sha256,
        "receipt_sha256": sha256_bytes(verifier_files["verifier_receipt.json"]),
        "receipt_size_bytes": len(verifier_files["verifier_receipt.json"]),
        "verifier_receipt_sha256": receipt["verifier_receipt_sha256"],
    }:
        raise EvidenceError("verifier COMMIT chain binding mismatch")
    return manifest, receipt, ledger


def _validate_checkpoint_boundary(payload: Any, *, require_self_hash: bool) -> None:
    expected_keys = set(CHECKPOINT_AUDIT_KEYS)
    if not require_self_hash:
        expected_keys.remove("audit_sha256")
    if type(payload) is not dict or set(payload) != expected_keys:
        raise EvidenceError("checkpoint audit schema changed")
    _exact_int(payload.get("schema_version"), 2, "checkpoint audit schema")
    if payload.get("checkpoint_id") != "PAPER_RESEARCH_JPY_ORACLE_CORRECTIVE_V2" \
            or payload.get("classification") != RUNTIME_CLASSIFICATION \
            or payload.get("anchor_status") != RUNTIME_ANCHOR_STATUS \
            or payload.get("holdout_state") != "UNOPENED" \
            or payload.get("superseded_checkpoint_commit") != SUPERSEDED_COMMIT \
            or payload.get("superseded_checkpoint_classification") \
                != "SUPERSEDED_NOT_ADMISSIBLE":
        raise EvidenceError("checkpoint identity/classification boundary mismatch")
    if payload.get("checkpoint_publication") != "EXCLUSIVE_HARDLINK_LOCAL_BUILDER" \
            or payload.get("checkpoint_commit_path") != CHECKPOINT_COMMIT_PATH \
            or payload.get("outer_launch_provenance_status") \
                != "CALLER_ASSERTED_PENDING_EXTERNAL_ANCHOR" \
            or payload.get("runtime_environment_scope") \
                != "HOST_LOCAL_NON_HERMETIC_STDLIB_NOT_BYTE_PINNED":
        raise EvidenceError("checkpoint local publication/provenance scope mismatch")
    for field, expected in (
        ("sealed_fd_execution", True),
        ("runtime_native_exclusive_publication", True),
        ("checkpoint_terminal_commit_required", True),
        ("release_evidence_eligible", False),
        ("local_reproducible_only", True),
        ("strategy_admission_eligible", False),
        ("producer_metrics_used", False),
        ("same_signal_ids_all_arms", True),
        ("all_proposals_have_all_arm_dispositions", True),
        ("official_strategy_run_performed", False),
        ("profit_evidence_generated", False),
        ("remote_anchor_verified", False),
        ("external_review_required_before_commit", True),
        ("pre_external_review_commit_push_allowed", False),
        ("legacy_seals_changed", False),
    ):
        _exact_bool(payload.get(field), expected, f"checkpoint audit {field}")
    _exact_int(
        payload.get("terminal_inventory_mtm_jpy_micros"), 0,
        "checkpoint audit terminal inventory MTM",
    )
    _exact_int(payload.get("external_orders"), 0, "checkpoint audit external orders")
    _exact_int(
        payload.get("legacy_official_oracle_pass_count"), 0,
        "checkpoint audit legacy Oracle passes",
    )
    _validate_exact_authority(payload.get("authority"), "checkpoint audit")
    source_hashes_value = payload.get("source_artifact_sha256")
    if type(source_hashes_value) is not dict \
            or set(source_hashes_value) != set(SOURCE_FILES):
        raise EvidenceError("checkpoint source hash set changed")
    for relative, value in source_hashes_value.items():
        _require_sha256(value, f"checkpoint source {relative}")
    evidence_hashes = payload.get("evidence_artifact_sha256")
    if type(evidence_hashes) is not dict \
            or set(evidence_hashes) != set(PRE_AUDIT_ARTIFACT_FILES) \
            or len(evidence_hashes) != EXPECTED_PRE_AUDIT_ARTIFACT_COUNT:
        raise EvidenceError("checkpoint pre-audit evidence hash set changed")
    for relative, value in evidence_hashes.items():
        if type(relative) is not str or not relative.startswith(f"{EVIDENCE_ROOT}/"):
            raise EvidenceError("checkpoint evidence hash path escaped its root")
        _require_sha256(value, f"checkpoint evidence {relative}")
    launcher_provenance = payload.get("launcher_runtime_provenance")
    launcher_provenance_keys = {
        "caller_asserted_bootstrap_source_sha256", "bootstrap_provenance",
        "pre_audit_capability_absence_proven", "interpreter_executable_sha256",
        "interpreter_identity_sha256", "interpreter_flags_sha256", "sys_path_sha256",
    }
    if type(launcher_provenance) is not dict \
            or set(launcher_provenance) != launcher_provenance_keys \
            or launcher_provenance.get("bootstrap_provenance") \
                != "PYTHON_C_NOT_SELF_AUTHENTICATING":
        raise EvidenceError("checkpoint launcher provenance schema mismatch")
    _exact_bool(
        launcher_provenance.get("pre_audit_capability_absence_proven"), False,
        "checkpoint launcher pre-audit capability proof",
    )
    for field in launcher_provenance_keys - {
        "bootstrap_provenance", "pre_audit_capability_absence_proven",
    }:
        _require_sha256(
            launcher_provenance.get(field), f"checkpoint launcher provenance {field}"
        )
    golden = payload.get("golden_reference")
    if type(golden) is not dict or set(golden) != {
        "fixture_id", "implementation_sha256", "expected_ledger_sha256",
        "expected_ledger_size_bytes", "expected_metrics_sha256",
        "sealed_oracle_ledger_exact_match", "sealed_oracle_metrics_exact_match",
        "independent_verifier_metrics_exact_match",
    } or golden.get("fixture_id") != "GOLDEN_USDJPY_LONG_V1":
        raise EvidenceError("checkpoint golden reference schema mismatch")
    for field in (
        "implementation_sha256", "expected_ledger_sha256", "expected_metrics_sha256",
    ):
        _require_sha256(golden.get(field), f"checkpoint golden {field}")
    if type(golden.get("expected_ledger_size_bytes")) is not int \
            or golden["expected_ledger_size_bytes"] <= 0:
        raise EvidenceError("checkpoint golden ledger size is invalid")
    for field in (
        "sealed_oracle_ledger_exact_match", "sealed_oracle_metrics_exact_match",
        "independent_verifier_metrics_exact_match",
    ):
        _exact_bool(golden.get(field), True, f"checkpoint golden {field}")
    for field in (
        "oracle_root_sha256", "verifier_receipt_sha256", "launcher_sha256",
        "legacy_coverage_sha256", "supersession_sha256",
    ):
        _require_sha256(payload.get(field), f"checkpoint audit {field}")
    if require_self_hash and payload.get("audit_sha256") \
            != embedded(payload, "audit_sha256"):
        raise EvidenceError("checkpoint audit self-hash mismatch")


def _validate_checkpoint_terminal_commit(
    payload: Any, expected_artifact_hashes: Mapping[str, str]
) -> None:
    if set(expected_artifact_hashes) != set(NONTERMINAL_ARTIFACT_FILES) \
            or len(expected_artifact_hashes) != EXPECTED_NONTERMINAL_ARTIFACT_COUNT:
        raise EvidenceError("checkpoint nonterminal artifact hash set changed")
    if type(payload) is not dict or set(payload) != set(CHECKPOINT_COMMIT_KEYS):
        raise EvidenceError("checkpoint terminal COMMIT schema changed")
    _exact_int(payload.get("schema_version"), 1, "checkpoint terminal COMMIT schema")
    _exact_int(
        payload.get("artifact_count"), len(expected_artifact_hashes),
        "checkpoint terminal COMMIT artifact count",
    )
    _exact_int(
        payload.get("external_orders"), 0,
        "checkpoint terminal COMMIT external orders",
    )
    _exact_bool(
        payload.get("strategy_admission_eligible"), False,
        "checkpoint terminal COMMIT strategy admission",
    )
    if payload.get("checkpoint_id") != "PAPER_RESEARCH_JPY_ORACLE_CORRECTIVE_V2" \
            or payload.get("classification") != RUNTIME_CLASSIFICATION \
            or payload.get("publication_state") != "TERMINAL_COMPLETE" \
            or payload.get("artifact_sha256") != dict(expected_artifact_hashes) \
            or payload.get("artifact_set_sha256") \
                != sha256_bytes(canonical(dict(expected_artifact_hashes))) \
            or payload.get("checkpoint_commit_sha256") \
                != embedded(payload, "checkpoint_commit_sha256"):
        raise EvidenceError("checkpoint terminal COMMIT binding mismatch")
    _require_sha256(payload.get("audit_sha256"), "checkpoint terminal audit")


def compute(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    supplied_root = Path(root)
    root_fd, root_identity = _open_root_anchor(supplied_root)
    try:
        root = supplied_root.resolve()
        after = os.stat(root)
        _require_safe_directory(after, "resolved checkpoint root")
        if _directory_identity(after) != root_identity:
            raise EvidenceError("checkpoint root changed while resolving")
    finally:
        os.close(root_fd)
    frozen_source_hashes = source_hashes(root)
    golden_payload = _load_golden_payload(
        root, frozen_source_hashes[GOLDEN_PATH]
    )
    coverage_before = legacy_coverage(root)
    superseded = supersession(root)
    with tempfile.TemporaryDirectory(prefix="qr-oracle-v2-sealed-") as temporary:
        scratch = Path(temporary)
        runtime_roots = {
            name: scratch / name
            for name in (
                "oracle_input_root",
                "oracle_output_root",
                "verifier_input_root",
                "verifier_output_root",
            )
        }
        for name, path in runtime_roots.items():
            _mkdir_private(path, name)
        _assert_distinct_private_roots(runtime_roots)
        oracle_input_root = runtime_roots["oracle_input_root"]
        oracle_output_root = runtime_roots["oracle_output_root"]
        verifier_input_root = runtime_roots["verifier_input_root"]
        verifier_output_root = runtime_roots["verifier_output_root"]
        oracle_request, oracle_request_path, golden_expected = fixture(
            oracle_input_root, golden_payload
        )
        oracle_launch = _invoke_launcher(
            root,
            oracle_input_root,
            oracle_output_root,
            "ORACLE",
            oracle_request_path,
            expected_launcher_sha256=frozen_source_hashes[LAUNCHER_PATH],
        )
        verifier_request_path, oracle_snapshot_paths = _verifier_request(
            root,
            oracle_input_root,
            oracle_output_root,
            verifier_input_root,
            oracle_request,
        )
        verifier_launch = _invoke_launcher(
            root,
            verifier_input_root,
            verifier_output_root,
            "VERIFIER",
            verifier_request_path,
            expected_launcher_sha256=frozen_source_hashes[LAUNCHER_PATH],
            trusted_oracle_paths=oracle_snapshot_paths,
        )

        _require_runtime_scratch_integrity(scratch, runtime_roots)
        manifest, receipt, ledger_bytes = _validate_inner_runtime_outputs(
            oracle_input_root,
            oracle_output_root,
            verifier_input_root,
            verifier_output_root,
            frozen_source_hashes,
            oracle_launch,
            verifier_launch,
        )
        if oracle_launch["launcher_sha256"] != verifier_launch["launcher_sha256"]:
            raise EvidenceError("Oracle and verifier used different launcher bytes")
        for field in (
            "caller_asserted_bootstrap_source_sha256",
            "bootstrap_provenance", "pre_audit_capability_absence_proven",
            "interpreter_executable_sha256",
            "interpreter_identity_sha256", "interpreter_flags_sha256", "sys_path_sha256",
        ):
            if oracle_launch[field] != verifier_launch[field]:
                raise EvidenceError(f"Oracle/verifier launcher {field} mismatch")
        if ledger_bytes != golden_expected["ledger_utf8"].encode("utf-8"):
            raise EvidenceError("sealed Oracle ledger differs from independent golden bytes")
        if manifest.get("oracle_metrics") != golden_expected["oracle_metrics"]:
            raise EvidenceError("sealed Oracle metrics differ from independent golden metrics")
        if receipt.get("verified_oracle_metrics") != golden_expected["oracle_metrics"]:
            raise EvidenceError("verifier metrics differ from independent golden metrics")

        coverage = legacy_coverage(root)
        if coverage != coverage_before:
            raise EvidenceError("legacy cycle coverage changed after freeze")
        _require_runtime_scratch_integrity(scratch, runtime_roots)
        artifacts = _collect_runtime_artifacts(
            oracle_input_root,
            oracle_output_root,
            verifier_input_root,
            verifier_output_root,
        )
        preterminal_manifest, preterminal_receipt, preterminal_ledger = (
            _validate_inner_runtime_outputs(
                oracle_input_root,
                oracle_output_root,
                verifier_input_root,
                verifier_output_root,
                frozen_source_hashes,
                oracle_launch,
                verifier_launch,
            )
        )
        if preterminal_manifest != manifest or preterminal_receipt != receipt \
                or preterminal_ledger != ledger_bytes:
            raise EvidenceError("inner runtime evidence changed at preterminal fence")
        artifacts[ORACLE_LAUNCH_RECEIPT_PATH] = canonical(oracle_launch) + b"\n"
        artifacts[VERIFIER_LAUNCH_RECEIPT_PATH] = canonical(verifier_launch) + b"\n"
        artifacts[LEGACY_COVERAGE_PATH] = canonical(coverage) + b"\n"
        artifacts[SUPERSESSION_PATH] = canonical(superseded) + b"\n"
        if set(artifacts) != set(PRE_AUDIT_ARTIFACT_FILES) \
                or len(artifacts) != EXPECTED_PRE_AUDIT_ARTIFACT_COUNT:
            raise EvidenceError("checkpoint pre-audit artifact set changed")

        metrics = manifest["oracle_metrics"]
        _assert_frozen_bindings(
            root,
            frozen_source_hashes,
            coverage_before["coverage_sha256"],
            superseded["supersession_sha256"],
            "after freeze",
        )
        audit = {
            "schema_version": 2,
            "checkpoint_id": "PAPER_RESEARCH_JPY_ORACLE_CORRECTIVE_V2",
            "classification": RUNTIME_CLASSIFICATION,
            "source_artifact_sha256": frozen_source_hashes,
            "evidence_artifact_sha256": {
                relative: sha256_bytes(value)
                for relative, value in sorted(artifacts.items())
            },
            "oracle_root_sha256": manifest["oracle_root_sha256"],
            "verifier_receipt_sha256": receipt["verifier_receipt_sha256"],
            "launcher_sha256": oracle_launch["launcher_sha256"],
            "launcher_runtime_provenance": {
                field: oracle_launch[field]
                for field in (
                    "caller_asserted_bootstrap_source_sha256",
                    "bootstrap_provenance", "pre_audit_capability_absence_proven",
                    "interpreter_executable_sha256",
                    "interpreter_identity_sha256", "interpreter_flags_sha256",
                    "sys_path_sha256",
                )
            },
            "golden_reference": {
                "fixture_id": "GOLDEN_USDJPY_LONG_V1",
                "implementation_sha256": frozen_source_hashes[GOLDEN_PATH],
                "expected_ledger_sha256": golden_expected["ledger_sha256"],
                "expected_ledger_size_bytes": golden_expected["ledger_size_bytes"],
                "expected_metrics_sha256": golden_expected["oracle_metrics"][
                    "metrics_sha256"
                ],
                "sealed_oracle_ledger_exact_match": True,
                "sealed_oracle_metrics_exact_match": True,
                "independent_verifier_metrics_exact_match": True,
            },
            "sealed_fd_execution": True,
            "runtime_native_exclusive_publication": True,
            "checkpoint_publication": "EXCLUSIVE_HARDLINK_LOCAL_BUILDER",
            "checkpoint_terminal_commit_required": True,
            "checkpoint_commit_path": CHECKPOINT_COMMIT_PATH,
            "release_evidence_eligible": False,
            "local_reproducible_only": True,
            "outer_launch_provenance_status": (
                "CALLER_ASSERTED_PENDING_EXTERNAL_ANCHOR"
            ),
            "runtime_environment_scope": (
                "HOST_LOCAL_NON_HERMETIC_STDLIB_NOT_BYTE_PINNED"
            ),
            "strategy_admission_eligible": False,
            "producer_metrics_used": False,
            "same_signal_ids_all_arms": metrics["same_signal_ids_all_arms"],
            "all_proposals_have_all_arm_dispositions": metrics[
                "all_proposals_have_all_arm_dispositions"
            ],
            "terminal_inventory_mtm_jpy_micros": metrics[
                "terminal_inventory_mtm_jpy_micros"
            ],
            "external_orders": metrics["external_orders"],
            "holdout_state": "UNOPENED",
            "official_strategy_run_performed": False,
            "profit_evidence_generated": False,
            "anchor_status": RUNTIME_ANCHOR_STATUS,
            "remote_anchor_verified": False,
            "external_review_required_before_commit": True,
            "pre_external_review_commit_push_allowed": False,
            "legacy_coverage_sha256": coverage["coverage_sha256"],
            "supersession_sha256": superseded["supersession_sha256"],
            "legacy_official_oracle_pass_count": 0,
            "legacy_seals_changed": False,
            "superseded_checkpoint_commit": SUPERSEDED_COMMIT,
            "superseded_checkpoint_classification": "SUPERSEDED_NOT_ADMISSIBLE",
            "authority": {
                "paper_only": True,
                "live_authority": False,
                "broker_account_access": False,
                "credential_access": False,
                "order_endpoint": False,
                "external_orders": 0,
                "deploy": False,
                "external_config_mutation": False,
            },
        }
        _validate_checkpoint_boundary(audit, require_self_hash=False)
        audit["audit_sha256"] = embedded(audit, "audit_sha256")
        _validate_checkpoint_boundary(audit, require_self_hash=True)
        artifacts[AUDIT_PATH] = canonical(audit) + b"\n"
        if set(artifacts) != set(NONTERMINAL_ARTIFACT_FILES) \
                or len(artifacts) != EXPECTED_NONTERMINAL_ARTIFACT_COUNT:
            raise EvidenceError("checkpoint nonterminal artifact set changed")
        checkpoint_artifact_hashes = {
            relative: sha256_bytes(value)
            for relative, value in sorted(artifacts.items())
        }
        checkpoint_commit = seal({
            "schema_version": 1,
            "checkpoint_id": audit["checkpoint_id"],
            "classification": audit["classification"],
            "audit_sha256": audit["audit_sha256"],
            "artifact_count": len(checkpoint_artifact_hashes),
            "artifact_sha256": checkpoint_artifact_hashes,
            "artifact_set_sha256": sha256_bytes(canonical(checkpoint_artifact_hashes)),
            "publication_state": "TERMINAL_COMPLETE",
            "strategy_admission_eligible": False,
            "external_orders": 0,
        }, "checkpoint_commit_sha256")
        _validate_checkpoint_terminal_commit(
            checkpoint_commit, checkpoint_artifact_hashes
        )
        artifacts[CHECKPOINT_COMMIT_PATH] = canonical(checkpoint_commit) + b"\n"
        if set(artifacts) != set(TOTAL_ARTIFACT_FILES) \
                or len(artifacts) != EXPECTED_TOTAL_ARTIFACT_COUNT:
            raise EvidenceError("checkpoint terminal artifact set changed")
        _assert_frozen_bindings(
            root,
            frozen_source_hashes,
            coverage_before["coverage_sha256"],
            superseded["supersession_sha256"],
            "after freeze",
        )
        _require_runtime_scratch_integrity(scratch, runtime_roots)
        final_manifest, final_receipt, final_ledger = _validate_inner_runtime_outputs(
            oracle_input_root,
            oracle_output_root,
            verifier_input_root,
            verifier_output_root,
            frozen_source_hashes,
            oracle_launch,
            verifier_launch,
        )
        if final_manifest != manifest or final_receipt != receipt \
                or final_ledger != ledger_bytes:
            raise EvidenceError("inner runtime evidence changed at final compute fence")
        return artifacts, audit


def _read_artifact_at(root_fd: int, relative: str) -> bytes:
    opened = _open_existing_parent_at(root_fd, _safe_relative(relative))
    if opened is None:
        raise EvidenceError(f"checkpoint evidence parent is missing: {relative}")
    parent_fd, name = opened
    try:
        return _read_regular_at(
            parent_fd, name, maximum_bytes=MAX_ARTIFACT_BYTES
        )
    finally:
        os.close(parent_fd)


def _evidence_file_set_at(root_fd: int, expected: set[str]) -> set[str]:
    allowed_directories: set[str] = set()
    for relative in expected:
        parts = _safe_relative(relative)
        for end in range(1, len(parts)):
            allowed_directories.add("/".join(parts[:end]))
    evidence_parts = _safe_relative(EVIDENCE_ROOT)
    descriptor = os.dup(root_fd)
    try:
        _safe_directory_fd(descriptor, "checkpoint evidence traversal root")
        traversed: list[str] = []
        for part in evidence_parts:
            try:
                named = os.stat(
                    part, dir_fd=descriptor, follow_symlinks=False
                )
                _require_safe_directory(
                    named, "checkpoint evidence directory pathname"
                )
                child = os.open(
                    part,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                return set()
            try:
                info = _safe_directory_fd(
                    child, "opened checkpoint evidence directory"
                )
            except EvidenceError:
                os.close(child)
                raise
            traversed.append(part)
            if "/".join(traversed) not in allowed_directories:
                os.close(child)
                raise EvidenceError("checkpoint evidence directory is unsafe")
            if _directory_identity(named) != _directory_identity(info):
                os.close(child)
                raise EvidenceError(
                    "checkpoint evidence directory changed while opening"
                )
            os.close(descriptor)
            descriptor = child

        result: set[str] = set()

        def walk(directory_fd: int, prefix: tuple[str, ...]) -> None:
            for name in sorted(os.listdir(directory_fd)):
                if type(name) is not str or name in {"", ".", ".."} or "/" in name:
                    raise EvidenceError("checkpoint evidence entry name is unsafe")
                relative = "/".join((*prefix, name))
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    _require_safe_directory(
                        info, f"checkpoint evidence directory {relative}"
                    )
                    if relative not in allowed_directories:
                        raise EvidenceError(
                            f"unexpected checkpoint evidence directory: {relative}"
                        )
                    child_fd = os.open(
                        name,
                        os.O_RDONLY
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=directory_fd,
                    )
                    try:
                        opened = _safe_directory_fd(
                            child_fd,
                            f"opened checkpoint evidence directory {relative}",
                        )
                        if _directory_identity(info) != _directory_identity(opened):
                            raise EvidenceError(
                                f"checkpoint evidence directory changed while opening: {relative}"
                            )
                        walk(child_fd, (*prefix, name))
                        after = _safe_directory_fd(
                            child_fd,
                            f"reread checkpoint evidence directory {relative}",
                        )
                        named = os.stat(
                            name, dir_fd=directory_fd, follow_symlinks=False
                        )
                        _require_safe_directory(
                            named,
                            f"checkpoint evidence directory pathname fence {relative}",
                        )
                        if _directory_identity(after) != _directory_identity(opened) \
                                or _directory_identity(named) \
                                != _directory_identity(opened):
                            raise EvidenceError(
                                f"checkpoint evidence directory changed during traversal: {relative}"
                            )
                    finally:
                        os.close(child_fd)
                elif stat.S_ISREG(info.st_mode):
                    _require_safe_regular(
                        info, f"checkpoint evidence file {relative}"
                    )
                    _read_regular_at(
                        directory_fd, name, maximum_bytes=MAX_ARTIFACT_BYTES
                    )
                    result.add(relative)
                else:
                    raise EvidenceError(f"unsafe checkpoint evidence entry: {relative}")

        walk(descriptor, evidence_parts)
        return result
    finally:
        os.close(descriptor)


def build(root: Path) -> dict[str, Any]:
    supplied_root = Path(root)
    root_fd, root_identity = _open_root_anchor(supplied_root)
    lock_fd = -1
    terminal_visible = False
    terminal_expected_bytes: bytes | None = None
    try:
        lock_fd = _acquire_builder_lock(root_fd)
        _assert_builder_lock_identity(root_fd, lock_fd)
        # Recover only already-existing builder-owned publication links before
        # compute traverses the evidence tree.  Existing-only traversal avoids
        # creating an evidence tree when source validation fails.
        for relative in sorted(TOTAL_ARTIFACT_FILES):
            _cleanup_existing_builder_partials_at(root_fd, relative)
        _assert_builder_lock_identity(root_fd, lock_fd)
        terminal_expected_bytes = _read_optional_artifact_at(
            root_fd, CHECKPOINT_COMMIT_PATH
        )
        terminal_visible = terminal_expected_bytes is not None
        try:
            artifacts, payload = compute(supplied_root)
            _validate_checkpoint_boundary(payload, require_self_hash=True)
            _assert_root_identity(supplied_root, root_identity, root_fd)
            _assert_builder_lock_identity(root_fd, lock_fd)
            expected_source_hashes = payload.get("source_artifact_sha256")
            if type(expected_source_hashes) is not dict:
                raise EvidenceError("checkpoint source binding is malformed")
            _assert_frozen_bindings(
                supplied_root,
                expected_source_hashes,
                payload["legacy_coverage_sha256"],
                payload["supersession_sha256"],
                "before publication",
            )
            _cleanup_all_builder_partials(root_fd, set(artifacts))
            _assert_builder_lock_identity(root_fd, lock_fd)
            existing = _evidence_file_set_at(root_fd, set(artifacts))
            terminal_visible = CHECKPOINT_COMMIT_PATH in existing
            unexpected = existing - set(artifacts)
            if unexpected:
                raise EvidenceError(
                    f"unexpected checkpoint evidence files: {sorted(unexpected)}"
                )
            publication_order = sorted(
                set(artifacts) - {AUDIT_PATH, CHECKPOINT_COMMIT_PATH}
            ) + [AUDIT_PATH, CHECKPOINT_COMMIT_PATH]
            for relative in publication_order:
                _assert_root_identity(supplied_root, root_identity, root_fd)
                _assert_builder_lock_identity(root_fd, lock_fd)
                if relative == CHECKPOINT_COMMIT_PATH:
                    terminal_expected_bytes = artifacts[relative]
                    _validate_checkpoint_boundary(payload, require_self_hash=True)
                    _assert_frozen_bindings(
                        supplied_root,
                        expected_source_hashes,
                        payload["legacy_coverage_sha256"],
                        payload["supersession_sha256"],
                        "before terminal commit",
                    )
                atomic_bytes_at(root_fd, relative, artifacts[relative])
                if relative == CHECKPOINT_COMMIT_PATH:
                    terminal_visible = True
                _assert_builder_lock_identity(root_fd, lock_fd)
                if relative == CHECKPOINT_COMMIT_PATH:
                    _assert_frozen_bindings(
                        supplied_root,
                        expected_source_hashes,
                        payload["legacy_coverage_sha256"],
                        payload["supersession_sha256"],
                        "after terminal commit",
                    )
            if _evidence_file_set_at(root_fd, set(artifacts)) != set(artifacts):
                raise EvidenceError("checkpoint evidence file set is incomplete")
            for relative, expected_bytes in sorted(artifacts.items()):
                if _read_artifact_at(root_fd, relative) != expected_bytes:
                    raise EvidenceError(
                        f"checkpoint evidence bytes changed before return: {relative}"
                    )
            published_audit = _canonical_json_object(
                _read_artifact_at(root_fd, AUDIT_PATH),
                "published checkpoint audit",
                CHECKPOINT_AUDIT_KEYS,
            )
            _validate_checkpoint_boundary(published_audit, require_self_hash=True)
            if published_audit != payload:
                raise EvidenceError("published checkpoint audit changed")
            commit = _canonical_json_object(
                _read_artifact_at(root_fd, CHECKPOINT_COMMIT_PATH),
                "published checkpoint terminal COMMIT",
                CHECKPOINT_COMMIT_KEYS,
            )
            expected_hashes = {
                relative: sha256_bytes(value)
                for relative, value in sorted(artifacts.items())
                if relative != CHECKPOINT_COMMIT_PATH
            }
            _validate_checkpoint_terminal_commit(commit, expected_hashes)
            if commit.get("audit_sha256") != payload["audit_sha256"]:
                raise EvidenceError("checkpoint terminal audit binding mismatch")
            _assert_root_identity(supplied_root, root_identity, root_fd)
            _assert_builder_lock_identity(root_fd, lock_fd)
            _assert_frozen_bindings(
                supplied_root,
                expected_source_hashes,
                payload["legacy_coverage_sha256"],
                payload["supersession_sha256"],
                "at final successful-return fence",
            )
            _validate_checkpoint_boundary(payload, require_self_hash=True)
            return payload
        except BaseException:
            current_terminal = _read_optional_artifact_at(
                root_fd, CHECKPOINT_COMMIT_PATH
            )
            terminal_visible = current_terminal is not None
            if terminal_visible:
                if terminal_expected_bytes is None:
                    raise EvidenceError(
                        "unbound terminal commit appeared during failed build"
                    )
                _invalidate_terminal_commit_at(root_fd, terminal_expected_bytes)
                terminal_visible = False
            raise
    finally:
        if lock_fd >= 0:
            lock_error: BaseException | None = None
            try:
                _assert_builder_lock_identity(root_fd, lock_fd)
            except BaseException as error:
                lock_error = error
                if terminal_visible:
                    try:
                        if terminal_expected_bytes is None:
                            raise EvidenceError(
                                "terminal commit lacks an invalidation byte binding"
                            )
                        _invalidate_terminal_commit_at(
                            root_fd, terminal_expected_bytes
                        )
                    except BaseException as invalidation_error:
                        cleanup_error = EvidenceError(
                            "builder lock changed and terminal commit invalidation failed"
                        )
                        cleanup_error.__cause__ = invalidation_error
                        lock_error = cleanup_error
                    finally:
                        terminal_visible = False
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            if lock_error is not None:
                os.close(root_fd)
                raise lock_error
        os.close(root_fd)


def validate(root: Path) -> dict[str, Any]:
    supplied_root = Path(root)
    root_fd, root_identity = _open_root_anchor(supplied_root)
    try:
        artifacts, payload = compute(supplied_root)
        _validate_checkpoint_boundary(payload, require_self_hash=True)
        _assert_root_identity(supplied_root, root_identity, root_fd)
        if _evidence_file_set_at(root_fd, set(artifacts)) != set(artifacts):
            raise EvidenceError("checkpoint evidence file set changed")
        for relative, expected in sorted(artifacts.items()):
            if _read_artifact_at(root_fd, relative) != expected:
                raise EvidenceError(f"oracle evidence changed: {relative}")
        published_audit = _canonical_json_object(
            _read_artifact_at(root_fd, AUDIT_PATH),
            "published checkpoint audit",
            CHECKPOINT_AUDIT_KEYS,
        )
        _validate_checkpoint_boundary(published_audit, require_self_hash=True)
        if published_audit != payload:
            raise EvidenceError("published checkpoint audit differs from recomputation")
        _assert_root_identity(supplied_root, root_identity, root_fd)
        return payload
    finally:
        os.close(root_fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    payload = build(args.root) if args.command == "build" else validate(args.root)
    print(json.dumps({
        "checkpoint_id": payload["checkpoint_id"],
        "classification": payload["classification"],
        "audit_sha256": payload["audit_sha256"],
        "oracle_root_sha256": payload["oracle_root_sha256"],
        "verifier_receipt_sha256": payload["verifier_receipt_sha256"],
        "launcher_sha256": payload["launcher_sha256"],
        "anchor_status": payload["anchor_status"],
        "external_review_required_before_commit": payload[
            "external_review_required_before_commit"
        ],
        "external_orders": payload["external_orders"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
