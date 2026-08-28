from __future__ import annotations

import ast
import base64
import errno
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import build_paper_research_jpy_oracle_corrective_v2 as checkpoint


ROOT = Path(__file__).resolve().parent
PYTHON = Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3")
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)

LEGACY_LIFECYCLE_STATE = "LEGACY_STALE_NON_ATTESTING"
CURRENT_LIFECYCLE_STATE = (
    "FUTURE_ONLY_ACCOUNTING_ONLY_LOCAL_UNANCHORED_NOT_ADMISSIBLE"
)
REFERENCE_RECEIPT_PATH = (
    f"{checkpoint.EVIDENCE_ROOT}/verifier_output/verifier_receipt.json"
)
EXPECTED_EVIDENCE_LIFECYCLE_POLICY = {
    "policy_kind": "IMMUTABLE_EVIDENCE_DERIVED_STATE_MACHINE",
    "state_receipt_chain": [
        REFERENCE_RECEIPT_PATH,
        checkpoint.AUDIT_PATH,
        checkpoint.CHECKPOINT_COMMIT_PATH,
    ],
    "state_selection": (
        "DERIVE_FROM_EXACT_RECEIPT_AUDIT_TERMINAL_BYTES_AND_CURRENT_VALIDATION_"
        "NEVER_FROM_CONTRACT_CURRENT_STATE_OR_REGENERATION_HISTORY"
    ),
    "exact_allowed_states": {
        LEGACY_LIFECYCLE_STATE: {
            "reference_result_sha256": (
                "ABSENT_FROM_RECEIPT_AND_VERIFIER_RELEASE_CONTENT_BINDING"
            ),
            "current_validation": "MUST_NOT_SUCCEED",
            "execution_attestation_eligible": False,
            "release_evidence_eligible": False,
            "admission_eligible": False,
        },
        CURRENT_LIFECYCLE_STATE: {
            "reference_result_sha256": (
                "PRESENT_AND_EXACT_BOUND_IN_RECEIPT_AND_VERIFIER_RELEASE_"
                "CONTENT_BINDING"
            ),
            "current_validation": "MUST_SUCCEED",
            "execution_attestation_eligible": False,
            "release_evidence_eligible": False,
            "admission_eligible": False,
        },
    },
    "mixed_partial_or_unknown_state": "REJECT",
}


def copy_inputs(destination: Path) -> None:
    destination.mkdir(parents=True, mode=0o700)
    git_directory = subprocess.run(
        ("/usr/bin/git", "-C", str(ROOT), "rev-parse", "--git-dir"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (destination / ".git").write_text(
        f"gitdir: {Path(git_directory).resolve()}\n", encoding="utf-8"
    )
    for relative in checkpoint.SOURCE_FILES:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for cycle in checkpoint.SEALED_CYCLES:
        relative = Path(f"evidence/orchestrator_state_v2/official_seal_v{cycle}.json")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for result_relative, ledger_relative in checkpoint.LEGACY_RUN_ARTIFACTS.values():
        for relative in (Path(result_relative), Path(ledger_relative)):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
    legacy_v1 = Path("evidence/paper_research_jpy_oracle_v1")
    for source in sorted((ROOT / legacy_v1).iterdir()):
        target = destination / legacy_v1 / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(
        ROOT / "V34_RESULT_VALIDATION_FAILURE.json",
        destination / "V34_RESULT_VALIDATION_FAILURE.json",
    )
    for relative in checkpoint.FAILED_CYCLE_RECEIPTS.values():
        shutil.copy2(ROOT / relative, destination / relative)


def run_builder(root: Path, command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(PYTHON),
            "-I",
            "-S",
            "-B",
            str(root / checkpoint.BUILDER_PATH),
            command,
            "--root",
            str(root),
        ],
        cwd=root,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _add_test_acl(path: Path, *, inherited: bool = False) -> None:
    rule = "everyone allow read"
    if inherited:
        rule += ",file_inherit,directory_inherit"
    subprocess.run(("/bin/chmod", "+a", rule, str(path)), check=True)


def _clear_test_acl(path: Path) -> None:
    subprocess.run(("/bin/chmod", "-N", str(path)), check=True)


def _add_test_xattr(path: Path) -> None:
    subprocess.run(
        (
            "/usr/bin/xattr",
            "-w",
            "com.quantrabbit.fixture",
            "safe",
            str(path),
        ),
        check=True,
    )


def build_copy(root: Path) -> dict:
    copy_inputs(root)
    completed = run_builder(root, "build")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stderr == ""
    response = json.loads(completed.stdout)
    assert response["classification"] == checkpoint.RUNTIME_CLASSIFICATION
    return json.loads((root / checkpoint.AUDIT_PATH).read_text(encoding="utf-8"))


def _canonical_lifecycle_object(raw: bytes, label: str) -> dict:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if type(value) is not dict or checkpoint.canonical(value) + b"\n" != raw:
        raise ValueError(f"{label} is not exact canonical object bytes")
    return value


def _exact_false(value: object, label: str) -> None:
    if type(value) is not bool or value is not False:
        raise ValueError(f"{label} must be exact false")


def _exact_sha256(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 \
            or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be exact lowercase sha256")
    return value


def _derive_evidence_lifecycle_state(
    audit_bytes: bytes,
    receipt_bytes: bytes,
    terminal_bytes: bytes,
    *,
    current_validation_succeeded: bool,
    expected_source_hashes: dict[str, str] | None = None,
) -> str:
    if type(current_validation_succeeded) is not bool:
        raise ValueError("current validation result must be exact bool")
    audit = _canonical_lifecycle_object(audit_bytes, "lifecycle audit")
    receipt = _canonical_lifecycle_object(receipt_bytes, "lifecycle receipt")
    terminal = _canonical_lifecycle_object(terminal_bytes, "lifecycle terminal")

    for value, key, label in (
        (audit, "audit_sha256", "lifecycle audit"),
        (receipt, "verifier_receipt_sha256", "lifecycle receipt"),
        (terminal, "checkpoint_commit_sha256", "lifecycle terminal"),
    ):
        if value.get(key) != checkpoint.embedded(value, key):
            raise ValueError(f"{label} self seal mismatch")
    if audit.get("verifier_receipt_sha256") \
            != receipt.get("verifier_receipt_sha256"):
        raise ValueError("lifecycle receipt is not bound by audit")
    evidence_hashes = audit.get("evidence_artifact_sha256")
    terminal_hashes = terminal.get("artifact_sha256")
    if type(evidence_hashes) is not dict or type(terminal_hashes) is not dict:
        raise ValueError("lifecycle hash maps must be exact objects")
    if evidence_hashes.get(REFERENCE_RECEIPT_PATH) \
            != checkpoint.sha256_bytes(receipt_bytes):
        raise ValueError("lifecycle receipt bytes are not bound by audit")
    if terminal.get("audit_sha256") != audit.get("audit_sha256") \
            or terminal_hashes.get(checkpoint.AUDIT_PATH) \
                != checkpoint.sha256_bytes(audit_bytes) \
            or terminal_hashes.get(REFERENCE_RECEIPT_PATH) \
                != checkpoint.sha256_bytes(receipt_bytes):
        raise ValueError("lifecycle audit/receipt bytes are not bound by terminal")
    if terminal.get("artifact_set_sha256") \
            != checkpoint.sha256_bytes(checkpoint.canonical(terminal_hashes)):
        raise ValueError("lifecycle terminal artifact set seal mismatch")

    _exact_false(
        audit.get("release_evidence_eligible"), "audit release eligibility"
    )
    _exact_false(
        audit.get("strategy_admission_eligible"), "audit admission eligibility"
    )
    _exact_false(audit.get("remote_anchor_verified"), "audit remote anchor")
    _exact_false(
        receipt.get("release_evidence_eligible"), "receipt release eligibility"
    )
    _exact_false(receipt.get("admission_eligible"), "receipt admission eligibility")
    _exact_false(
        terminal.get("strategy_admission_eligible"),
        "terminal admission eligibility",
    )

    binding = receipt.get("verifier_release_content_binding")
    receipt_has_reference = "reference_result_sha256" in receipt
    binding_has_reference = type(binding) is dict \
        and "reference_result_sha256" in binding
    if receipt_has_reference != binding_has_reference:
        raise ValueError("partial reference-result lifecycle binding")

    if not receipt_has_reference:
        if current_validation_succeeded:
            raise ValueError("legacy lifecycle cannot pass current validation")
        if binding is not None:
            raise ValueError("legacy lifecycle has unexpected release binding")
        if audit.get("classification") \
                != "ACCOUNTING_ONLY_LOCAL_REPRODUCIBLE_NOT_ADMISSIBLE" \
                or receipt.get("classification") != "ACCOUNTING_ONLY_NOT_ADMISSIBLE" \
                or terminal.get("classification") \
                    != "ACCOUNTING_ONLY_LOCAL_REPRODUCIBLE_NOT_ADMISSIBLE" \
                or audit.get("anchor_status") \
                    != "LOCAL_REPRODUCIBLE_ACCOUNTING_ONLY" \
                or receipt.get("anchor_status") \
                    != "LOCAL_REPRODUCIBLE_ACCOUNTING_ONLY" \
                or set(audit.get("source_artifact_sha256", ())) \
                    == set(checkpoint.SOURCE_FILES):
            raise ValueError("unknown reference-free lifecycle state")
        return LEGACY_LIFECYCLE_STATE

    reference_result_sha256 = _exact_sha256(
        receipt.get("reference_result_sha256"), "reference result"
    )
    if binding.get("reference_result_sha256") != reference_result_sha256:
        raise ValueError("reference-result lifecycle binding mismatch")
    if not current_validation_succeeded:
        raise ValueError("current lifecycle requires successful validation")
    if expected_source_hashes is None:
        raise ValueError("current lifecycle source hashes are required")
    if audit.get("classification") != CURRENT_LIFECYCLE_STATE \
            or receipt.get("classification") != CURRENT_LIFECYCLE_STATE \
            or terminal.get("classification") != CURRENT_LIFECYCLE_STATE \
            or audit.get("anchor_status") != "LOCAL_UNANCHORED" \
            or receipt.get("anchor_status") != "LOCAL_UNANCHORED" \
            or audit.get("source_artifact_sha256") != expected_source_hashes \
            or set(evidence_hashes) != set(checkpoint.PRE_AUDIT_ARTIFACT_FILES) \
            or terminal.get("artifact_count") \
                != checkpoint.EXPECTED_NONTERMINAL_ARTIFACT_COUNT \
            or set(terminal_hashes) \
                != set(checkpoint.NONTERMINAL_ARTIFACT_FILES):
        raise ValueError("unknown current lifecycle state")
    return CURRENT_LIFECYCLE_STATE


def _reseal_lifecycle_chain(
    audit: dict, receipt: dict, terminal: dict
) -> tuple[bytes, bytes, bytes]:
    receipt["verifier_receipt_sha256"] = checkpoint.embedded(
        receipt, "verifier_receipt_sha256"
    )
    receipt_bytes = checkpoint.canonical(receipt) + b"\n"
    audit["verifier_receipt_sha256"] = receipt["verifier_receipt_sha256"]
    audit["evidence_artifact_sha256"][REFERENCE_RECEIPT_PATH] = (
        checkpoint.sha256_bytes(receipt_bytes)
    )
    audit["audit_sha256"] = checkpoint.embedded(audit, "audit_sha256")
    audit_bytes = checkpoint.canonical(audit) + b"\n"
    terminal["audit_sha256"] = audit["audit_sha256"]
    terminal["artifact_sha256"][REFERENCE_RECEIPT_PATH] = (
        checkpoint.sha256_bytes(receipt_bytes)
    )
    terminal["artifact_sha256"][checkpoint.AUDIT_PATH] = (
        checkpoint.sha256_bytes(audit_bytes)
    )
    terminal["artifact_set_sha256"] = checkpoint.sha256_bytes(
        checkpoint.canonical(terminal["artifact_sha256"])
    )
    terminal["checkpoint_commit_sha256"] = checkpoint.embedded(
        terminal, "checkpoint_commit_sha256"
    )
    return (
        audit_bytes,
        receipt_bytes,
        checkpoint.canonical(terminal) + b"\n",
    )


def _literal_legacy_lifecycle_fixture() -> tuple[bytes, bytes, bytes]:
    """Return a deterministic historical chain without reading evidence files."""
    receipt = {
        "classification": "ACCOUNTING_ONLY_NOT_ADMISSIBLE",
        "anchor_status": "LOCAL_REPRODUCIBLE_ACCOUNTING_ONLY",
        "release_evidence_eligible": False,
        "admission_eligible": False,
        "verifier_release_content_binding": None,
        "verifier_receipt_sha256": "0" * 64,
    }
    audit = {
        "classification": "ACCOUNTING_ONLY_LOCAL_REPRODUCIBLE_NOT_ADMISSIBLE",
        "anchor_status": "LOCAL_REPRODUCIBLE_ACCOUNTING_ONLY",
        "release_evidence_eligible": False,
        "strategy_admission_eligible": False,
        "remote_anchor_verified": False,
        "source_artifact_sha256": {
            "historical_fixture_v1.py": checkpoint.sha256_bytes(
                b"immutable historical lifecycle fixture\n"
            ),
        },
        "evidence_artifact_sha256": {REFERENCE_RECEIPT_PATH: "0" * 64},
        "verifier_receipt_sha256": "0" * 64,
        "audit_sha256": "0" * 64,
    }
    terminal = {
        "classification": "ACCOUNTING_ONLY_LOCAL_REPRODUCIBLE_NOT_ADMISSIBLE",
        "strategy_admission_eligible": False,
        "artifact_count": 2,
        "artifact_sha256": {
            REFERENCE_RECEIPT_PATH: "0" * 64,
            checkpoint.AUDIT_PATH: "0" * 64,
        },
        "audit_sha256": "0" * 64,
        "artifact_set_sha256": "0" * 64,
        "checkpoint_commit_sha256": "0" * 64,
    }
    chain = _reseal_lifecycle_chain(audit, receipt, terminal)
    expected = (
        (706, "91cff385c9d4c995f835e53b77189ffa9e1e571947cb46db90245a0f81f97186"),
        (299, "cf618982fed203048369d35584eada17fbe9987049ab6a6c5e5de37808bcc609"),
        (711, "fe4ca77d011acc92102a8d32f3c2e68cc08f9079f080496630df150d666090eb"),
    )
    if tuple(
        (len(raw), checkpoint.sha256_bytes(raw)) for raw in chain
    ) != expected:
        raise AssertionError("literal legacy lifecycle fixture bytes drifted")
    return chain


def _synthetic_current_lifecycle_fixture(
    source_hashes: dict[str, str],
) -> tuple[bytes, bytes, bytes]:
    reference_hash = checkpoint.sha256_bytes(
        b"immutable synthetic current reference result\n"
    )
    receipt = {
        "classification": CURRENT_LIFECYCLE_STATE,
        "anchor_status": "LOCAL_UNANCHORED",
        "release_evidence_eligible": False,
        "admission_eligible": False,
        "reference_result_sha256": reference_hash,
        "verifier_release_content_binding": {
            "reference_result_sha256": reference_hash,
        },
        "verifier_receipt_sha256": "0" * 64,
    }
    audit = {
        "classification": CURRENT_LIFECYCLE_STATE,
        "anchor_status": "LOCAL_UNANCHORED",
        "release_evidence_eligible": False,
        "strategy_admission_eligible": False,
        "remote_anchor_verified": False,
        "source_artifact_sha256": dict(source_hashes),
        "evidence_artifact_sha256": {
            relative: checkpoint.sha256_bytes(
                f"synthetic pre-audit:{relative}\n".encode("utf-8")
            )
            for relative in checkpoint.PRE_AUDIT_ARTIFACT_FILES
        },
        "verifier_receipt_sha256": "0" * 64,
        "audit_sha256": "0" * 64,
    }
    terminal = {
        "classification": CURRENT_LIFECYCLE_STATE,
        "strategy_admission_eligible": False,
        "artifact_count": checkpoint.EXPECTED_NONTERMINAL_ARTIFACT_COUNT,
        "artifact_sha256": {
            relative: checkpoint.sha256_bytes(
                f"synthetic nonterminal:{relative}\n".encode("utf-8")
            )
            for relative in checkpoint.NONTERMINAL_ARTIFACT_FILES
        },
        "audit_sha256": "0" * 64,
        "artifact_set_sha256": "0" * 64,
        "checkpoint_commit_sha256": "0" * 64,
    }
    return _reseal_lifecycle_chain(audit, receipt, terminal)


def _source_hashes_only(source_root: Path) -> dict[str, str]:
    return {
        relative: checkpoint.sha256_bytes((source_root / relative).read_bytes())
        for relative in checkpoint.SOURCE_FILES
    }


def _source_only_lifecycle_observation(
    source_root: Path,
) -> bytes:
    """Observe policy and both states using SOURCE_FILES, never evidence state."""
    contract_path = checkpoint.CONTRACT_PATH
    if contract_path not in checkpoint.SOURCE_FILES:
        raise AssertionError("lifecycle contract is outside SOURCE_FILES")
    contract = json.loads((source_root / contract_path).read_bytes())
    source_hashes = _source_hashes_only(source_root)
    legacy_chain = _literal_legacy_lifecycle_fixture()
    current_chain = _synthetic_current_lifecycle_fixture(source_hashes)
    observation = {
        "policy": contract["evidence_lifecycle_policy"],
        "legacy": _derive_evidence_lifecycle_state(
            *legacy_chain,
            current_validation_succeeded=False,
            expected_source_hashes=source_hashes,
        ),
        "current": _derive_evidence_lifecycle_state(
            *current_chain,
            current_validation_succeeded=True,
            expected_source_hashes=source_hashes,
        ),
    }
    return checkpoint.canonical(observation)


def assert_boundary(payload: dict) -> None:
    assert payload["checkpoint_id"] == "PAPER_RESEARCH_JPY_ORACLE_CORRECTIVE_V2"
    assert payload["classification"] == checkpoint.RUNTIME_CLASSIFICATION
    assert payload["sealed_fd_execution"] is True
    assert payload["runtime_native_exclusive_publication"] is True
    assert payload["checkpoint_publication"] == "EXCLUSIVE_HARDLINK_LOCAL_BUILDER"
    assert payload["checkpoint_terminal_commit_required"] is True
    assert payload["checkpoint_commit_path"] == checkpoint.CHECKPOINT_COMMIT_PATH
    assert payload["release_evidence_eligible"] is False
    assert payload["local_reproducible_only"] is True
    assert payload["outer_launch_provenance_status"] == (
        "CALLER_ASSERTED_PENDING_EXTERNAL_ANCHOR"
    )
    assert payload["runtime_environment_scope"] == (
        "HOST_LOCAL_NON_HERMETIC_STDLIB_NOT_BYTE_PINNED"
    )
    assert payload["strategy_admission_eligible"] is False
    assert payload["producer_metrics_used"] is False
    assert payload["same_signal_ids_all_arms"] is True
    assert payload["all_proposals_have_all_arm_dispositions"] is True
    assert payload["terminal_inventory_mtm_jpy_micros"] == 0
    assert payload["external_orders"] == 0
    assert payload["holdout_state"] == "UNOPENED"
    assert payload["official_strategy_run_performed"] is False
    assert payload["profit_evidence_generated"] is False
    assert payload["anchor_status"] == checkpoint.RUNTIME_ANCHOR_STATUS
    assert payload["remote_anchor_verified"] is False
    assert payload["external_review_required_before_commit"] is True
    assert payload["pre_external_review_commit_push_allowed"] is False
    assert payload["legacy_official_oracle_pass_count"] == 0
    assert payload["legacy_seals_changed"] is False
    assert len(payload["supersession_sha256"]) == 64
    assert payload["superseded_checkpoint_commit"] == checkpoint.SUPERSEDED_COMMIT
    assert payload["superseded_checkpoint_classification"] == (
        "SUPERSEDED_NOT_ADMISSIBLE"
    )
    assert payload["authority"] == {
        "paper_only": True,
        "live_authority": False,
        "broker_account_access": False,
        "credential_access": False,
        "order_endpoint": False,
        "external_orders": 0,
        "deploy": False,
        "external_config_mutation": False,
    }
    assert payload["audit_sha256"] == checkpoint.embedded(payload, "audit_sha256")


def evidence_snapshot(root: Path) -> dict[str, tuple[str, int, int]]:
    evidence = root / checkpoint.EVIDENCE_ROOT
    return {
        path.relative_to(root).as_posix(): (
            checkpoint.sha256_file(path),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in sorted(evidence.rglob("*"))
        if path.is_file()
    }


def test_builder_is_deterministic_read_only_on_validate_and_tamper_evident(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    first = build_copy(first_root)
    assert_boundary(first)
    before = evidence_snapshot(first_root)

    second_build = run_builder(first_root, "build")
    assert second_build.returncode == 0, second_build.stdout + second_build.stderr
    assert evidence_snapshot(first_root) == before

    validation = run_builder(first_root, "validate")
    assert validation.returncode == 0, validation.stdout + validation.stderr
    assert evidence_snapshot(first_root) == before

    second_root = tmp_path / "second"
    second = build_copy(second_root)
    assert first == second
    assert {
        key: (value[0], value[1]) for key, value in evidence_snapshot(first_root).items()
    } == {
        key: (value[0], value[1]) for key, value in evidence_snapshot(second_root).items()
    }

    ledger = first_root / checkpoint.EVIDENCE_ROOT / "oracle_output/oracle_ledger.jsonl"
    ledger.write_bytes(ledger.read_bytes() + b"\n")
    rejected = run_builder(first_root, "validate")
    assert rejected.returncode != 0
    assert "oracle evidence changed" in rejected.stderr


@pytest.mark.parametrize("mode", (0o777, 0o775))
def test_world_or_group_writable_root_rejects_before_first_build(
    tmp_path: Path,
    mode: int,
) -> None:
    root = tmp_path / f"unsafe-root-{mode:o}"
    copy_inputs(root)
    root.chmod(mode)
    rejected = run_builder(root, "build")
    assert rejected.returncode != 0
    assert "owner-controlled directory" in rejected.stderr
    assert not (root / checkpoint.CHECKPOINT_COMMIT_PATH).exists()


@pytest.mark.parametrize("relative", (checkpoint.ORACLE_PATH, checkpoint.BUILDER_LOCK_PATH))
@pytest.mark.parametrize("mode", (0o666, 0o660))
def test_world_or_group_writable_source_file_rejects_before_build(
    tmp_path: Path,
    relative: str,
    mode: int,
) -> None:
    root = tmp_path / f"unsafe-source-{Path(relative).name}-{mode:o}"
    copy_inputs(root)
    (root / relative).chmod(mode)
    rejected = run_builder(root, "build")
    assert rejected.returncode != 0
    assert "owner-controlled regular file" in rejected.stderr
    assert not (root / checkpoint.CHECKPOINT_COMMIT_PATH).exists()


@pytest.mark.parametrize(
    ("relative", "mode"),
    (
        (checkpoint.EVIDENCE_ROOT, 0o777),
        (checkpoint.EVIDENCE_ROOT, 0o775),
        (f"{checkpoint.EVIDENCE_ROOT}/inputs", 0o777),
        (f"{checkpoint.EVIDENCE_ROOT}/inputs", 0o775),
        (f"{checkpoint.EVIDENCE_ROOT}/inputs/source_manifest.json", 0o666),
        (f"{checkpoint.EVIDENCE_ROOT}/inputs/source_manifest.json", 0o660),
        (checkpoint.AUDIT_PATH, 0o666),
        (checkpoint.AUDIT_PATH, 0o660),
        (checkpoint.CHECKPOINT_COMMIT_PATH, 0o666),
        (checkpoint.CHECKPOINT_COMMIT_PATH, 0o660),
    ),
)
def test_unsafe_published_directory_or_file_rejects_validate_and_rebuild(
    tmp_path: Path,
    relative: str,
    mode: int,
) -> None:
    root = tmp_path / f"unsafe-published-{mode:o}-{Path(relative).name}"
    build_copy(root)
    target = root / relative
    target.chmod(mode)

    validation = run_builder(root, "validate")
    assert validation.returncode != 0
    assert "owner-controlled" in validation.stderr

    rebuild = run_builder(root, "build")
    assert rebuild.returncode != 0
    assert "owner-controlled" in rebuild.stderr


def test_repository_modes_remain_permitted_and_generated_tree_is_private(
    tmp_path: Path,
) -> None:
    root = tmp_path / "safe-owner-modes"
    copy_inputs(root)
    root.chmod(0o755)
    for relative in checkpoint.SOURCE_FILES:
        (root / relative).chmod(0o644)

    built = run_builder(root, "build")
    assert built.returncode == 0, built.stdout + built.stderr
    validated = run_builder(root, "validate")
    assert validated.returncode == 0, validated.stdout + validated.stderr

    evidence_root = root / checkpoint.EVIDENCE_ROOT
    assert stat.S_IMODE(evidence_root.stat().st_mode) == 0o700
    for path in evidence_root.rglob("*"):
        expected_mode = 0o700 if path.is_dir() else 0o600
        assert stat.S_IMODE(path.stat().st_mode) == expected_mode

    for path in evidence_root.rglob("*"):
        if path.is_file():
            path.chmod(0o644)
    for path in sorted(evidence_root.rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(0o755)
    evidence_root.chmod(0o755)
    validated_safe_outer_modes = run_builder(root, "validate")
    assert validated_safe_outer_modes.returncode == 0, (
        validated_safe_outer_modes.stdout + validated_safe_outer_modes.stderr
    )
    idempotent_safe_outer_modes = run_builder(root, "build")
    assert idempotent_safe_outer_modes.returncode == 0, (
        idempotent_safe_outer_modes.stdout + idempotent_safe_outer_modes.stderr
    )


@pytest.mark.parametrize(
    "relative",
    (".", checkpoint.ORACLE_PATH, checkpoint.BUILDER_LOCK_PATH),
)
def test_builder_rejects_extended_acl_before_first_build(
    tmp_path: Path,
    relative: str,
) -> None:
    root = tmp_path / f"acl-prebuild-{Path(relative).name}"
    copy_inputs(root)
    target = root if relative == "." else root / relative
    _add_test_acl(target)
    rejected = run_builder(root, "build")
    assert rejected.returncode != 0
    assert "extended ACL" in rejected.stderr
    assert not (root / checkpoint.CHECKPOINT_COMMIT_PATH).exists()


def test_builder_acl_api_fails_closed_on_unsupported_and_api_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeCall:
        def __init__(self, result: object, error_number: int = 0) -> None:
            self.result = result
            self.error_number = error_number
            self.calls = 0

        def __call__(self, *args: object) -> object:
            self.calls += 1
            checkpoint.ctypes.set_errno(self.error_number)
            return self.result

    class FakeLibrary:
        def __init__(self, get: FakeCall, free: FakeCall) -> None:
            self.acl_get_fd_np = get
            self.acl_free = free

    with pytest.raises(checkpoint.EvidenceError, match="unavailable on this host"):
        checkpoint._bind_extended_acl_api("linux", object())
    with pytest.raises(checkpoint.EvidenceError, match="API is unavailable"):
        checkpoint._bind_extended_acl_api("darwin", object())

    path = tmp_path / "acl-api-file"
    path.write_bytes(b"x")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        get = FakeCall(None, errno.EOPNOTSUPP)
        free = FakeCall(0)
        monkeypatch.setattr(checkpoint, "_ACL_GET_FD_NP", get)
        monkeypatch.setattr(checkpoint, "_ACL_FREE", free)
        with pytest.raises(checkpoint.EvidenceError, match="inspection failed"):
            checkpoint._require_no_extended_acl_fd(descriptor, "fake ACL")

        get = FakeCall(1)
        free = FakeCall(-1, errno.EIO)
        monkeypatch.setattr(checkpoint, "_ACL_GET_FD_NP", get)
        monkeypatch.setattr(checkpoint, "_ACL_FREE", free)
        with pytest.raises(checkpoint.EvidenceError, match="release failed"):
            checkpoint._require_no_extended_acl_fd(descriptor, "fake ACL")
        assert free.calls == 1
    finally:
        os.close(descriptor)


def test_builder_rejects_acl_on_every_outer_evidence_role_and_partial(
    tmp_path: Path,
) -> None:
    root = tmp_path / "acl-published"
    build_copy(root)
    targets = (
        root / checkpoint.EVIDENCE_ROOT,
        root / checkpoint.EVIDENCE_ROOT / "inputs",
        root / checkpoint.EVIDENCE_ROOT / "inputs/source_manifest.json",
        root / checkpoint.AUDIT_PATH,
        root / checkpoint.CHECKPOINT_COMMIT_PATH,
        root / checkpoint.BUILDER_LOCK_PATH,
    )
    for target in targets:
        _add_test_acl(target)
        try:
            validation = run_builder(root, "validate")
            assert validation.returncode != 0
            assert "extended ACL" in validation.stderr
            rebuild = run_builder(root, "build")
            assert rebuild.returncode != 0
            assert "extended ACL" in rebuild.stderr
        finally:
            _clear_test_acl(target)
        restored = run_builder(root, "build")
        assert restored.returncode == 0, restored.stdout + restored.stderr

    terminal = root / checkpoint.CHECKPOINT_COMMIT_PATH
    partial = terminal.parent / (
        f".{terminal.name}.builder-partial-acl-fixture"
    )
    partial.write_bytes(b"acl partial fixture\n")
    partial.chmod(0o600)
    _add_test_acl(partial)
    rejected = run_builder(root, "build")
    assert rejected.returncode != 0
    assert "extended ACL" in rejected.stderr
    assert partial.exists()
    _clear_test_acl(partial)
    partial.unlink()


def test_builder_rejects_inherited_acl_and_allows_plain_xattrs(
    tmp_path: Path,
) -> None:
    inherited_root = tmp_path / "acl-inherited"
    copy_inputs(inherited_root)
    evidence_parent = inherited_root / "evidence"
    evidence_parent.mkdir(exist_ok=True, mode=0o700)
    _add_test_acl(evidence_parent, inherited=True)
    rejected = run_builder(inherited_root, "build")
    assert rejected.returncode != 0
    assert "extended ACL" in rejected.stderr

    xattr_root = tmp_path / "xattr-safe"
    copy_inputs(xattr_root)
    for target in (
        xattr_root,
        xattr_root / checkpoint.ORACLE_PATH,
        xattr_root / checkpoint.BUILDER_LOCK_PATH,
    ):
        _add_test_xattr(target)
    built = run_builder(xattr_root, "build")
    assert built.returncode == 0, built.stdout + built.stderr
    for target in (
        xattr_root / checkpoint.EVIDENCE_ROOT,
        xattr_root / checkpoint.AUDIT_PATH,
        xattr_root / checkpoint.CHECKPOINT_COMMIT_PATH,
    ):
        _add_test_xattr(target)
    validated = run_builder(xattr_root, "validate")
    assert validated.returncode == 0, validated.stdout + validated.stderr


def test_checkpoint_binds_sealed_launcher_golden_and_exact_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "copy"
    payload = build_copy(root)
    assert_boundary(payload)
    evidence = root / checkpoint.EVIDENCE_ROOT
    manifest = json.loads(
        (evidence / "oracle_output/oracle_manifest.json").read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (evidence / "verifier_output/verifier_receipt.json").read_text(encoding="utf-8")
    )
    oracle_launch = json.loads(
        (root / checkpoint.ORACLE_LAUNCH_RECEIPT_PATH).read_text(encoding="utf-8")
    )
    verifier_launch = json.loads(
        (root / checkpoint.VERIFIER_LAUNCH_RECEIPT_PATH).read_text(encoding="utf-8")
    )
    assert oracle_launch["snapshot_mode"] == "SEALED_FD_COMPILE_EXEC_V2"
    assert verifier_launch["snapshot_mode"] == "SEALED_FD_COMPILE_EXEC_V2"
    assert oracle_launch["release_evidence_eligible"] is False
    assert verifier_launch["release_evidence_eligible"] is False
    assert oracle_launch["local_reproducible_only"] is True
    assert verifier_launch["local_reproducible_only"] is True
    assert oracle_launch["bootstrap_provenance"] == "PYTHON_C_NOT_SELF_AUTHENTICATING"
    assert verifier_launch["bootstrap_provenance"] == "PYTHON_C_NOT_SELF_AUTHENTICATING"
    assert oracle_launch["pre_audit_capability_absence_proven"] is False
    assert verifier_launch["pre_audit_capability_absence_proven"] is False
    assert oracle_launch["outer_launch_provenance_status"] == (
        "CALLER_ASSERTED_PENDING_EXTERNAL_ANCHOR"
    )
    assert verifier_launch["outer_launch_provenance_status"] == (
        "CALLER_ASSERTED_PENDING_EXTERNAL_ANCHOR"
    )
    assert oracle_launch["runtime_environment_scope"] == (
        "HOST_LOCAL_NON_HERMETIC_STDLIB_NOT_BYTE_PINNED"
    )
    assert verifier_launch["runtime_environment_scope"] == (
        "HOST_LOCAL_NON_HERMETIC_STDLIB_NOT_BYTE_PINNED"
    )
    assert oracle_launch["launcher_sha256"] == verifier_launch["launcher_sha256"]
    assert oracle_launch["launcher_sha256"] == payload["launcher_sha256"]
    assert payload["launcher_runtime_provenance"] == {
        field: oracle_launch[field]
        for field in (
            "caller_asserted_bootstrap_source_sha256",
            "bootstrap_provenance",
            "pre_audit_capability_absence_proven",
            "interpreter_executable_sha256",
            "interpreter_identity_sha256",
            "interpreter_flags_sha256",
            "sys_path_sha256",
        )
    }
    assert manifest["oracle_release_content_binding"]["launcher_sha256"] == payload[
        "launcher_sha256"
    ]
    assert receipt["oracle_release_content_binding"] == manifest[
        "oracle_release_content_binding"
    ]
    assert receipt["verifier_release_content_binding"]["launcher_sha256"] == payload[
        "launcher_sha256"
    ]
    reference_code = root / checkpoint.REFERENCE_PATH
    reference_contract = root / checkpoint.REFERENCE_CONTRACT_PATH
    reference_code_snapshot = evidence / "inputs/reference_code_snapshot.py"
    reference_contract_snapshot = evidence / "inputs/reference_contract_snapshot.json"
    assert reference_code_snapshot.read_bytes() == reference_code.read_bytes()
    assert reference_contract_snapshot.read_bytes() == reference_contract.read_bytes()
    verifier_request = json.loads(
        (evidence / "inputs/verifier_request.json").read_text(encoding="utf-8")
    )
    assert verifier_request["reference_code_snapshot"] == {
        "artifact_id": "reference_code_snapshot",
        "relative_path": "inputs/reference_code_snapshot.py",
        "sha256": checkpoint.sha256_file(reference_code),
        "size_bytes": reference_code.stat().st_size,
    }
    assert verifier_request["reference_contract_snapshot"] == {
        "artifact_id": "reference_contract_snapshot",
        "relative_path": "inputs/reference_contract_snapshot.json",
        "sha256": checkpoint.sha256_file(reference_contract),
        "size_bytes": reference_contract.stat().st_size,
    }
    assert receipt["reference_engine_id"] == checkpoint.REFERENCE_ENGINE_ID
    assert receipt["reference_code_sha256"] == checkpoint.sha256_file(reference_code)
    assert receipt["reference_contract_sha256"] == checkpoint.sha256_file(
        reference_contract
    )
    assert receipt["reference_journal_transaction_count"] == 22
    assert receipt["reference_all_transactions_balanced"] is True
    assert receipt["reference_accounting_diagnostics_only"] is True
    assert receipt["reference_n_eff_statistical_admission_allowed"] is False
    assert receipt["reference_direction_accuracy_profit_gate_allowed"] is False
    assert receipt["verifier_release_content_binding"][
        "reference_code_sha256"
    ] == checkpoint.sha256_file(reference_code)
    assert receipt["verifier_release_content_binding"][
        "reference_contract_sha256"
    ] == checkpoint.sha256_file(reference_contract)
    expected_reference_input_root = checkpoint.sha256_bytes(checkpoint.canonical({
        "artifact_sha256": {
            label: receipt["input_artifact_sha256"][label]
            for label in sorted(checkpoint.ORACLE_INPUT_LABELS)
        }
    }))
    assert receipt["reference_input_root_sha256"] == expected_reference_input_root
    ledger_bytes = (evidence / "oracle_output/oracle_ledger.jsonl").read_bytes()
    reference_projection = {
        "all_transactions_balanced": True,
        "engine_id": checkpoint.REFERENCE_ENGINE_ID,
        "input_root_sha256": expected_reference_input_root,
        "journal_root_sha256": receipt["reference_journal_root_sha256"],
        "journal_transaction_count": 22,
        "ledger_row_count": len(ledger_bytes.splitlines()),
        "ledger_sha256": checkpoint.sha256_bytes(ledger_bytes),
        "ledger_terminal_hash": manifest["oracle_ledger_terminal_hash"],
        "oracle_metrics_sha256": manifest["oracle_metrics"]["metrics_sha256"],
        "proposal_provenance_root_sha256": manifest[
            "proposal_provenance_root_sha256"
        ],
    }
    expected_reference_projection_sha256 = checkpoint.sha256_bytes(
        checkpoint.canonical(reference_projection)
    )
    assert receipt["reference_economic_projection_sha256"] == (
        expected_reference_projection_sha256
    )
    reference_result_snapshot = {
        "engine_id": checkpoint.REFERENCE_ENGINE_ID,
        "input_root_sha256": expected_reference_input_root,
        "ledger_bytes_base64": base64.b64encode(ledger_bytes).decode("ascii"),
        "ledger_row_count": len(ledger_bytes.splitlines()),
        "ledger_terminal_hash": manifest["oracle_ledger_terminal_hash"],
        "oracle_metrics": manifest["oracle_metrics"],
        "proposal_provenance_root_sha256": manifest[
            "proposal_provenance_root_sha256"
        ],
        "journal_root_sha256": receipt["reference_journal_root_sha256"],
        "journal_transaction_count": 22,
        "all_transactions_balanced": True,
        "economic_projection_sha256": expected_reference_projection_sha256,
    }
    expected_reference_result_sha256 = checkpoint.sha256_bytes(
        checkpoint.canonical(reference_result_snapshot) + b"\n"
    )
    assert receipt["reference_result_sha256"] == expected_reference_result_sha256
    assert receipt["verifier_release_content_binding"][
        "reference_result_sha256"
    ] == expected_reference_result_sha256
    assert manifest["oracle_execution_provenance_scope"] == (
        checkpoint.EXECUTION_PROVENANCE_SCOPE
    )
    assert receipt["verifier_execution_provenance_scope"] == (
        checkpoint.EXECUTION_PROVENANCE_SCOPE
    )
    assert manifest["classification"] == checkpoint.RUNTIME_CLASSIFICATION
    assert receipt["classification"] == checkpoint.RUNTIME_CLASSIFICATION
    assert manifest["causal_signal_admission"] is False
    assert receipt["admission_eligible"] is False
    golden_expected = checkpoint._load_golden_payload(
        root, checkpoint.sha256_file(root / checkpoint.GOLDEN_PATH)
    )["expected"]
    assert payload["golden_reference"] == {
        "fixture_id": "GOLDEN_USDJPY_LONG_V1",
        "implementation_sha256": checkpoint.sha256_file(root / checkpoint.GOLDEN_PATH),
        "expected_ledger_sha256": golden_expected["ledger_sha256"],
        "expected_ledger_size_bytes": golden_expected["ledger_size_bytes"],
        "expected_metrics_sha256": golden_expected["oracle_metrics"][
            "metrics_sha256"
        ],
        "sealed_oracle_ledger_exact_match": True,
        "sealed_oracle_metrics_exact_match": True,
        "independent_verifier_metrics_exact_match": True,
    }
    assert payload["golden_reference"]["implementation_sha256"] == (
        payload["source_artifact_sha256"][checkpoint.GOLDEN_PATH]
    )
    assert set(payload["source_artifact_sha256"]) == set(checkpoint.SOURCE_FILES)
    assert payload["source_artifact_sha256"][
        checkpoint.ORACLE_FINANCE_TEST_PATH
    ] == checkpoint.sha256_file(root / checkpoint.ORACLE_FINANCE_TEST_PATH)
    for relative in (
        checkpoint.REFERENCE_PATH,
        checkpoint.REFERENCE_CONTRACT_PATH,
        checkpoint.REFERENCE_TEST_PATH,
        checkpoint.REFERENCE_MUTATION_TEST_PATH,
    ):
        assert payload["source_artifact_sha256"][relative] == checkpoint.sha256_file(
            root / relative
        )
    assert (
        len(checkpoint.RUNTIME_ARTIFACT_FILES)
        == checkpoint.EXPECTED_RUNTIME_ARTIFACT_COUNT
        == 22
    )
    assert set(payload["evidence_artifact_sha256"]) == set(
        checkpoint.PRE_AUDIT_ARTIFACT_FILES
    )
    assert (
        len(payload["evidence_artifact_sha256"])
        == checkpoint.EXPECTED_PRE_AUDIT_ARTIFACT_COUNT
        == 26
    )
    for relative, expected in payload["evidence_artifact_sha256"].items():
        assert checkpoint.sha256_file(root / relative) == expected
    terminal_commit = json.loads(
        (root / checkpoint.CHECKPOINT_COMMIT_PATH).read_text(encoding="utf-8")
    )
    assert terminal_commit["publication_state"] == "TERMINAL_COMPLETE"
    assert terminal_commit["audit_sha256"] == payload["audit_sha256"]
    assert (
        terminal_commit["artifact_count"]
        == checkpoint.EXPECTED_NONTERMINAL_ARTIFACT_COUNT
        == 27
    )
    assert set(terminal_commit["artifact_sha256"]) == set(
        checkpoint.NONTERMINAL_ARTIFACT_FILES
    )
    assert terminal_commit["checkpoint_commit_sha256"] == checkpoint.embedded(
        terminal_commit, "checkpoint_commit_sha256"
    )
    assert set(evidence_snapshot(root)) == set(checkpoint.TOTAL_ARTIFACT_FILES)
    assert len(evidence_snapshot(root)) == checkpoint.EXPECTED_TOTAL_ARTIFACT_COUNT == 28
    names = {
        path.name
        for path in evidence.rglob("*")
        if path.is_file()
    }
    assert not any(name.startswith(".oracle-") or name.startswith(".verifier-") for name in names)


def test_reference_diagnostic_contract_and_exact_tree_are_non_admissible() -> None:
    contract = json.loads(
        (ROOT / checkpoint.CONTRACT_PATH).read_text(encoding="utf-8")
    )
    reference = contract["double_entry_reference_diagnostics"]
    assert reference["engine_id"] == checkpoint.REFERENCE_ENGINE_ID
    assert reference["role"] == "INDEPENDENT_DOUBLE_ENTRY_ACCOUNTING_DIAGNOSTIC_ONLY"
    assert reference["code_source"] == checkpoint.REFERENCE_PATH
    assert reference["contract_source"] == checkpoint.REFERENCE_CONTRACT_PATH
    assert reference["frozen_source_files"] == [
        checkpoint.REFERENCE_PATH,
        checkpoint.REFERENCE_CONTRACT_PATH,
        checkpoint.REFERENCE_TEST_PATH,
        checkpoint.REFERENCE_MUTATION_TEST_PATH,
    ]
    assert reference["verifier_request_descriptors"] == {
        "reference_code_snapshot": "inputs/reference_code_snapshot.py",
        "reference_contract_snapshot": "inputs/reference_contract_snapshot.json",
    }
    assert reference["launcher_fd_options"] == {
        "reference_code": "--reference-code-fd",
        "reference_contract": "--reference-contract-fd",
    }
    assert reference["launcher_fd_scope"] == (
        "REFERENCE_CODE_AND_CONTRACT_FDS_TERMINATE_AT_THE_SEALED_LAUNCHER_"
        "AND_ARE_NOT_FORWARDED_TO_THE_VERIFIER"
    )
    assert reference["receipt_fields"] == [
        "reference_engine_id",
        "reference_code_sha256",
        "reference_contract_sha256",
        "reference_input_root_sha256",
        "reference_journal_root_sha256",
        "reference_journal_transaction_count",
        "reference_all_transactions_balanced",
        "reference_economic_projection_sha256",
        "reference_result_sha256",
        "reference_accounting_diagnostics_only",
        "reference_n_eff_statistical_admission_allowed",
        "reference_direction_accuracy_profit_gate_allowed",
    ]
    assert reference["expected_golden_fixture_journal_transaction_count"] == 22
    assert reference["accounting_diagnostics_only"] is True
    assert reference["causal_signal_admission"] is False
    assert reference["release_evidence_eligible"] is False
    assert reference["admission_status"] == "NON_ADMISSIBLE"
    assert reference["n_eff_statistical_admission_allowed"] is False
    assert reference["direction_accuracy_profit_gate_allowed"] is False
    assert reference["profitability_or_admission_override_allowed"] is False
    assert reference["content_binding"] == {
        "all_four_reference_sources_in_checkpoint_source_hash_set": True,
        "request_snapshot_hashes_equal_frozen_source_hashes": True,
        "launcher_sealed_fd_bytes_equal_request_snapshot_bytes": True,
        "receipt_reference_hashes_equal_frozen_source_hashes": True,
        "verifier_release_content_binding_includes_reference_hashes": True,
        "reference_projection_hash_bound_to_input_journal_ledger_metrics_and_provenance": True,
        "receipt_and_verifier_release_binding_include_exact_reference_result_sha256": True,
    }
    assert all(
        type(value) is bool and value is True
        for value in reference["content_binding"].values()
    )
    assert reference["reference_result_canonicalization"] == {
        "exact_key_count": 11,
        "ledger_bytes_encoding": (
            "STRICT_STANDARD_PADDED_ASCII_BASE64_WITH_CANONICAL_ROUND_TRIP"
        ),
        "encoding": (
            "UTF8_RFC8259_CANONICAL_SORTED_KEYS_NO_INSIGNIFICANT_WHITESPACE_"
            "SINGLE_TERMINAL_LF"
        ),
        "sha256_recipe": (
            "SHA256_OF_THE_EXACT_CANONICAL_REFERENCE_RESULT_BYTES_INCLUDING_"
            "THE_TERMINAL_LF"
        ),
        "persistence": "IN_MEMORY_ONLY_NO_ADDITIONAL_RUNTIME_OR_EVIDENCE_ARTIFACT",
    }
    assert reference["phase_3_execution_boundary"] == {
        "reference_replay_owner": "SEALED_FD_LAUNCHER",
        "reference_input": (
            "EXACT_NINE_RAW_ARTIFACT_BYTE_STRINGS_FROM_AN_IMMUTABLE_SORTED_"
            "TUPLE_BUNDLE"
        ),
        "verifier_receives_only_immutable_sorted_tuple_bundles_reference_result_bytes_and_hashes": True,
        "verifier_live_callable_received": False,
        "verifier_module_received": False,
        "verifier_file_descriptor_received": False,
        "verifier_filesystem_path_received": False,
        "verifier_write_closure_received": False,
        "verifier_rename_closure_received": False,
        "verifier_returns": (
            "CANONICAL_RECEIPT_AND_COMMIT_BYTES_TO_THE_SEALED_LAUNCHER"
        ),
        "publication_owner": "SEALED_FD_LAUNCHER_VALIDATES_THEN_PUBLISHES",
        "additional_reference_result_artifact_written": False,
    }
    verification = contract["verification"]
    assert verification["double_entry_reference_pair_pinned_in_launcher"] is True
    assert verification["double_entry_reference_replays_same_nine_raw_input_artifacts"] \
        is True
    assert verification["double_entry_reference_result_role"] == (
        "DIAGNOSTICS_ONLY_NON_ADMISSIBLE"
    )
    assert verification["verifier_recomputes_expected_canonical_ledger_bytes"] \
        is False
    assert verification[
        "launcher_precomputes_canonical_reference_result_from_exact_nine_raw_inputs"
    ] is True
    assert verification[
        "verifier_validates_canonical_reference_result_and_exact_oracle_match"
    ] is True
    assert verification["content_binding_claim"] == (
        "LOCAL_EXACT_BYTES_HASH_CHAIN_RELEASE_PAIR_AND_CANONICAL_REFERENCE_"
        "RESULT_BINDING_NOT_EXECUTION_ATTESTATION"
    )
    assert verification["profit_and_admission_read_oracle_metrics_only"] is True
    assert verification["exact_checkpoint_artifact_counts"] == {
        "sealed_runtime": 22,
        "pre_audit": 26,
        "nonterminal": 27,
        "total_with_terminal_commit": 28,
    }
    assert {
        "inputs/reference_code_snapshot.py",
        "inputs/reference_contract_snapshot.json",
    }.issubset(checkpoint.VERIFIER_INPUT_FILES)
    assert checkpoint.RUNTIME_ARTIFACT_FILES == frozenset({
        *checkpoint.ORACLE_INPUT_FILES,
        "inputs/oracle_code_snapshot.py",
        "inputs/oracle_contract_snapshot.json",
        "inputs/oracle_schema_snapshot.json",
        "inputs/reference_code_snapshot.py",
        "inputs/reference_contract_snapshot.json",
        "inputs/verifier_request.json",
        *(f"oracle_output/{name}" for name in checkpoint.ORACLE_OUTPUT_FILES),
        *(f"verifier_output/{name}" for name in checkpoint.VERIFIER_OUTPUT_FILES),
    })
    assert len(checkpoint.RUNTIME_ARTIFACT_FILES) == 22
    assert len(checkpoint.PRE_AUDIT_ARTIFACT_FILES) == 26
    assert len(checkpoint.NONTERMINAL_ARTIFACT_FILES) == 27
    assert len(checkpoint.TOTAL_ARTIFACT_FILES) == 28
    assert contract["evidence_lifecycle_policy"] \
        == EXPECTED_EVIDENCE_LIFECYCLE_POLICY
    assert "checked_in_evidence_state" not in contract


def test_reference_result_snapshot_hash_includes_terminal_lf_and_strict_base64() -> None:
    ledger = b'{"row":1}\n'
    snapshot = {
        "engine_id": checkpoint.REFERENCE_ENGINE_ID,
        "input_root_sha256": "1" * 64,
        "ledger_bytes_base64": base64.b64encode(ledger).decode("ascii"),
        "ledger_row_count": 1,
        "ledger_terminal_hash": "2" * 64,
        "oracle_metrics": {},
        "proposal_provenance_root_sha256": "3" * 64,
        "journal_root_sha256": "4" * 64,
        "journal_transaction_count": 22,
        "all_transactions_balanced": True,
        "economic_projection_sha256": "5" * 64,
    }
    expected = checkpoint.sha256_bytes(checkpoint.canonical(snapshot) + b"\n")
    assert checkpoint._reference_result_snapshot_sha256(snapshot, ledger) == expected
    assert expected != checkpoint.sha256_bytes(checkpoint.canonical(snapshot))

    extra = dict(snapshot, unknown=True)
    with pytest.raises(checkpoint.EvidenceError, match="snapshot schema changed"):
        checkpoint._reference_result_snapshot_sha256(extra, ledger)

    unpadded = dict(snapshot)
    unpadded["ledger_bytes_base64"] = snapshot["ledger_bytes_base64"].rstrip("=")
    with pytest.raises(checkpoint.EvidenceError, match="base64 is invalid"):
        checkpoint._reference_result_snapshot_sha256(unpadded, ledger)


def test_legacy_coverage_and_v1_supersession_are_sidecars_only(tmp_path: Path) -> None:
    root = tmp_path / "copy"
    copy_inputs(root)
    legacy_before, aggregate_before = checkpoint._legacy_frozen_artifact_set(root)
    assert len(legacy_before) == checkpoint.LEGACY_FROZEN_ARTIFACT_COUNT == 45
    assert aggregate_before == checkpoint.LEGACY_FROZEN_ARTIFACT_STREAM_SHA256
    completed = run_builder(root, "build")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    legacy_after, aggregate_after = checkpoint._legacy_frozen_artifact_set(root)
    assert legacy_after == legacy_before
    assert aggregate_after == aggregate_before
    coverage = json.loads((root / checkpoint.LEGACY_COVERAGE_PATH).read_text())
    assert coverage["sealed_cycle_count"] == 13
    assert coverage["invalid_cycle_count"] == 1
    assert coverage["execution_failure_cycle_count"] == 3
    assert coverage["official_oracle_pass_count"] == 0
    assert coverage["legacy_seals_changed"] is False
    assert {row["oracle_input_coverage"] for row in coverage["cycles"]} == {"MISSING"}
    assert next(row for row in coverage["cycles"] if row["cycle"] == "V34")[
        "coverage_state"
    ] == "LEGACY_INVALID_NOT_ADMISSIBLE"
    superseded = json.loads((root / checkpoint.SUPERSESSION_PATH).read_text())
    assert superseded["superseded_commit"] == checkpoint.SUPERSEDED_COMMIT
    assert superseded["classification"] == "SUPERSEDED_NOT_ADMISSIBLE"
    assert superseded["retroactive_promotion_allowed"] is False
    assert superseded["strategy_or_profit_evidence"] is False
    assert superseded["legacy_v1_evidence_binding"]["file_count"] == 12
    assert superseded["legacy_v1_evidence_binding"]["aggregate_sha256"] == (
        checkpoint.SUPERSEDED_V1_AGGREGATE_SHA256
    )
    assert superseded["legacy_v1_evidence_binding"][
        "prior_embedded_audit_sha256"
    ] == checkpoint.SUPERSEDED_V1_AUDIT_SHA256
    assert superseded["legacy_v1_evidence_binding"]["git_binding"] == {
        "object_database_verified": True,
        "commit": checkpoint.SUPERSEDED_COMMIT,
        "commit_tree": checkpoint.SUPERSEDED_COMMIT_TREE,
        "subtree_path": checkpoint.SUPERSEDED_V1_GIT_PATH,
        "subtree": checkpoint.SUPERSEDED_V1_SUBTREE,
        "file_mode": "100644",
        "blob_oids": superseded["legacy_v1_evidence_binding"]["git_binding"][
            "blob_oids"
        ],
    }
    assert set(
        superseded["legacy_v1_evidence_binding"]["git_binding"]["blob_oids"]
    ) == set(checkpoint.LEGACY_V1_EVIDENCE_NAMES)


@pytest.mark.parametrize(
    "failure",
    (
        "missing_v25",
        "unexpected_v34",
        "invalid_v25_payload",
        "invalid_v34_failure_payload",
        "tampered_v25_result",
        "tampered_v41_ledger",
        "tampered_v34_result",
        "dangling_v26_seal",
        "nested_seal_parent_symlink",
        "nested_v1_parent_symlink",
        "unexpected_v26_result",
    ),
)
def test_legacy_state_mismatch_fails_before_evidence_publication(
    tmp_path: Path,
    failure: str,
) -> None:
    root = tmp_path / failure
    copy_inputs(root)
    if failure == "missing_v25":
        (root / "evidence/orchestrator_state_v2/official_seal_v25.json").unlink()
    elif failure == "unexpected_v34":
        target = root / "evidence/orchestrator_state_v2/official_seal_v34.json"
        target.write_bytes(b"{}\n")
    elif failure == "invalid_v25_payload":
        target = root / "evidence/orchestrator_state_v2/official_seal_v25.json"
        target.write_bytes(b"{}\n")
    elif failure == "invalid_v34_failure_payload":
        (root / "V34_RESULT_VALIDATION_FAILURE.json").write_bytes(b"{}\n")
    elif failure == "tampered_v25_result":
        relative = checkpoint.LEGACY_RUN_ARTIFACTS[25][0]
        (root / relative).write_bytes((root / relative).read_bytes() + b"\n")
    elif failure == "tampered_v41_ledger":
        relative = checkpoint.LEGACY_RUN_ARTIFACTS[41][1]
        (root / relative).write_bytes((root / relative).read_bytes() + b"\n")
    elif failure == "tampered_v34_result":
        relative = checkpoint.LEGACY_RUN_ARTIFACTS[34][0]
        (root / relative).write_bytes((root / relative).read_bytes() + b"\n")
    elif failure == "dangling_v26_seal":
        target = root / "evidence/orchestrator_state_v2/official_seal_v26.json"
        target.symlink_to(root / "does-not-exist.json")
    elif failure == "nested_seal_parent_symlink":
        source = root / "evidence/orchestrator_state_v2"
        outside = tmp_path / "outside-orchestrator-state"
        source.rename(outside)
        source.symlink_to(outside, target_is_directory=True)
    elif failure == "nested_v1_parent_symlink":
        source = root / "evidence/paper_research_jpy_oracle_v1"
        outside = tmp_path / "outside-v1-evidence"
        source.rename(outside)
        source.symlink_to(outside, target_is_directory=True)
    else:
        unexpected = root / "evidence/run_unexpected_v26_official_001"
        unexpected.mkdir()
        (unexpected / "result_unexpected_v26.json").write_bytes(b"{}\n")
    completed = run_builder(root, "build")
    assert completed.returncode != 0
    assert not (root / checkpoint.EVIDENCE_ROOT).exists()


def test_joint_legacy_reseal_cannot_replace_the_frozen_artifact_set(
    tmp_path: Path,
) -> None:
    root = tmp_path / "joint-reseal"
    copy_inputs(root)
    result_relative, ledger_relative = checkpoint.LEGACY_RUN_ARTIFACTS[25]
    ledger_path = root / ledger_relative
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    rows[0]["diagnostics"]["owner_tamper_probe"] = True
    ledger_path.write_bytes(b"".join(checkpoint.canonical(row) + b"\n" for row in rows))

    result_path = root / result_relative
    result = json.loads(result_path.read_text())
    result["proposal_ledger_sha256"] = checkpoint.sha256_file(ledger_path)
    result["result_sha256"] = checkpoint.embedded(result, "result_sha256")
    result_path.write_bytes(checkpoint.canonical(result) + b"\n")

    seal_path = root / "evidence/orchestrator_state_v2/official_seal_v25.json"
    legacy_seal = json.loads(seal_path.read_text())
    legacy_seal["ledger_sha256"] = checkpoint.sha256_file(ledger_path)
    legacy_seal["result_file_sha256"] = checkpoint.sha256_file(result_path)
    legacy_seal["embedded_result_sha256"] = result["result_sha256"]
    legacy_seal["official_seal_sha256"] = checkpoint.embedded(
        legacy_seal, "official_seal_sha256"
    )
    seal_path.write_bytes(checkpoint.canonical(legacy_seal) + b"\n")

    with pytest.raises(
        checkpoint.EvidenceError, match="legacy frozen artifact set hash mismatch"
    ):
        checkpoint.build(root)
    assert not (root / checkpoint.EVIDENCE_ROOT).exists()


def test_builder_has_no_oracle_verifier_import_or_arbitrary_runtime_surface() -> None:
    source = (ROOT / checkpoint.BUILDER_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "paper_research_jpy_oracle_v2" not in imports
    assert "paper_research_oracle_verifier_v2" not in imports
    assert "paper_research_fd_launcher_v2" not in imports
    assert "paper_research_double_entry_reference_v2" not in imports
    assert "--executable" not in source
    assert "--argv" not in source
    parser_choices = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {"ORACLE", "VERIFIER"}.issubset(parser_choices)


def test_launcher_artifact_open_failure_closes_prior_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    checkpoint._mkdir_private(input_root, "test input root")
    checkpoint._mkdir_private(output_root, "test output root")
    original = checkpoint._open_root_relative_readonly_regular
    opened: list[int] = []

    def fail_after_first(root_fd: int, relative: str, label: str) -> int:
        if opened:
            raise OSError("injected artifact open failure")
        descriptor = original(root_fd, relative, label)
        opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(
        checkpoint, "_open_root_relative_readonly_regular", fail_after_first
    )
    with pytest.raises(OSError, match="injected artifact open failure"):
        checkpoint._invoke_launcher(
            ROOT,
            input_root,
            output_root,
            "ORACLE",
            ROOT / checkpoint.CONTRACT_PATH,
            expected_launcher_sha256=checkpoint.sha256_file(
                ROOT / checkpoint.LAUNCHER_PATH
            ),
        )
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_launcher_bootstrap_failure_closes_every_open_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    checkpoint._mkdir_private(input_root, "test input root")
    checkpoint._mkdir_private(output_root, "test output root")
    original_open = os.open
    opened: list[int] = []
    launcher_sha256 = checkpoint.sha256_file(ROOT / checkpoint.LAUNCHER_PATH)

    def tracked_open(*args: object, **kwargs: object) -> int:
        descriptor = original_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def fail_bootstrap(path: Path, expected_sha256: str) -> tuple[str, str]:
        del path, expected_sha256
        raise checkpoint.EvidenceError("injected bootstrap failure")

    monkeypatch.setattr(checkpoint.os, "open", tracked_open)
    monkeypatch.setattr(checkpoint, "_fixed_bootstrap_from_launcher", fail_bootstrap)
    with pytest.raises(checkpoint.EvidenceError, match="injected bootstrap failure"):
        checkpoint._invoke_launcher(
            ROOT,
            input_root,
            output_root,
            "ORACLE",
            ROOT / checkpoint.CONTRACT_PATH,
            expected_launcher_sha256=launcher_sha256,
        )
    assert len(opened) == 10
    for descriptor in opened:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_compute_uses_four_distinct_private_root_inodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "four-runtime-roots"
    copy_inputs(root)
    original_invoke = checkpoint._invoke_launcher
    observed: dict[str, tuple[tuple[int, int], tuple[int, int], str, str]] = {}

    def record_roots(
        source_root: Path,
        input_root: Path,
        output_root: Path,
        operation: str,
        request_path: Path,
        **kwargs: object,
    ) -> dict:
        input_info = os.lstat(input_root)
        output_info = os.lstat(output_root)
        observed[operation] = (
            (input_info.st_dev, input_info.st_ino),
            (output_info.st_dev, output_info.st_ino),
            input_root.name,
            output_root.name,
        )
        return original_invoke(
            source_root,
            input_root,
            output_root,
            operation,
            request_path,
            **kwargs,
        )

    monkeypatch.setattr(checkpoint, "_invoke_launcher", record_roots)
    checkpoint.compute(root)

    assert set(observed) == {"ORACLE", "VERIFIER"}
    assert observed["ORACLE"][2:] == (
        "oracle_input_root",
        "oracle_output_root",
    )
    assert observed["VERIFIER"][2:] == (
        "verifier_input_root",
        "verifier_output_root",
    )
    identities = {
        observed[operation][index]
        for operation in ("ORACLE", "VERIFIER")
        for index in (0, 1)
    }
    assert len(identities) == 4


def test_private_copy_is_inode_distinct_and_collision_fails(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    checkpoint._mkdir_private(source_root, "copy source root")
    checkpoint._mkdir_private(target_root, "copy target root")
    source = checkpoint._write_private_bytes(
        source_root / "source.bin", b"sealed-copy-bytes", "copy source"
    )
    target = target_root / "target.bin"

    copied = checkpoint._copy_private_regular(
        source,
        target,
        "test private copy",
        maximum_bytes=64,
    )
    source_info = os.lstat(source)
    copied_info = os.lstat(copied)
    assert copied.read_bytes() == source.read_bytes() == b"sealed-copy-bytes"
    assert (source_info.st_dev, source_info.st_ino) != (
        copied_info.st_dev,
        copied_info.st_ino,
    )
    assert source_info.st_nlink == copied_info.st_nlink == 1

    with pytest.raises(FileExistsError):
        checkpoint._copy_private_regular(
            source,
            target,
            "test private copy collision",
            maximum_bytes=64,
        )


def test_private_copy_fd_reread_fence_rejects_source_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    checkpoint._mkdir_private(source_root, "copy source root")
    checkpoint._mkdir_private(target_root, "copy target root")
    source = checkpoint._write_private_bytes(
        source_root / "source.bin", b"before-copy", "copy source"
    )
    original_write = checkpoint._write_private_bytes

    def write_then_mutate(path: Path, value: bytes, label: str) -> Path:
        result = original_write(path, value, label)
        source.write_bytes(b"after-copy")
        return result

    monkeypatch.setattr(checkpoint, "_write_private_bytes", write_then_mutate)
    with pytest.raises(
        checkpoint.EvidenceError,
        match="copy bytes changed across fd reread fence",
    ):
        checkpoint._copy_private_regular(
            source,
            target_root / "target.bin",
            "test fenced copy",
            maximum_bytes=64,
        )


def test_bounded_regular_read_rejects_oversized_file(tmp_path: Path) -> None:
    source_root = tmp_path / "bounded"
    checkpoint._mkdir_private(source_root, "bounded read root")
    source = checkpoint._write_private_bytes(
        source_root / "source.bin", b"too-large", "bounded read source"
    )
    with pytest.raises(checkpoint.EvidenceError, match="exceeds its byte bound"):
        checkpoint.read_regular_bytes(
            source,
            "bounded test read",
            maximum_bytes=len(b"too-large") - 1,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("extra_file", "file set changed"),
        ("extra_directory", "directory set changed"),
        ("missing_reference_code", "file set changed"),
        ("missing_reference_contract", "file set changed"),
        ("nested_directory_acl", "extended ACL"),
    ),
)
def test_exact_scratch_tree_rejects_extra_or_missing_entries(
    tmp_path: Path, mutation: str, message: str
) -> None:
    scratch = tmp_path / f"scratch-{mutation}"
    checkpoint._mkdir_private(scratch, "test scratch root")
    roots = {
        name: scratch / name
        for name in (
            "oracle_input_root",
            "oracle_output_root",
            "verifier_input_root",
            "verifier_output_root",
        )
    }
    expected_by_root = {
        "oracle_input_root": checkpoint.ORACLE_INPUT_FILES,
        "oracle_output_root": checkpoint.ORACLE_OUTPUT_ROOT_FILES,
        "verifier_input_root": checkpoint.VERIFIER_INPUT_FILES,
        "verifier_output_root": checkpoint.VERIFIER_OUTPUT_ROOT_FILES,
    }
    for name, root in roots.items():
        checkpoint._mkdir_private(root, f"test {name}")
        for relative in sorted(expected_by_root[name]):
            target = root / relative
            if target.parent != root and not target.parent.exists():
                checkpoint._mkdir_private(
                    target.parent, f"test {name} {target.parent.name}"
                )
            checkpoint._write_private_bytes(
                target, b"", f"test {name} {relative}"
            )
    checkpoint._assert_exact_scratch_tree(scratch, roots)

    if mutation == "extra_file":
        checkpoint._write_private_bytes(
            roots["verifier_output_root"] / "extra.bin",
            b"extra",
            "unexpected scratch file",
        )
    elif mutation == "extra_directory":
        checkpoint._mkdir_private(
            roots["verifier_output_root"] / "extra",
            "unexpected scratch directory",
        )
    elif mutation == "missing_reference_code":
        (roots["verifier_input_root"] / "inputs/reference_code_snapshot.py").unlink()
    elif mutation == "missing_reference_contract":
        (
            roots["verifier_input_root"]
            / "inputs/reference_contract_snapshot.json"
        ).unlink()
    else:
        _add_test_acl(roots["verifier_input_root"] / "inputs")
    with pytest.raises(checkpoint.EvidenceError, match=message):
        checkpoint._assert_exact_scratch_tree(scratch, roots)


@pytest.mark.parametrize(
    "target_role",
    (
        "oracle_input_root",
        "oracle_output_root",
        "verifier_input_root",
        "verifier_output_root",
    ),
)
def test_runtime_scratch_integrity_rejects_nested_acl_added_after_launchers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_role: str,
) -> None:
    root = tmp_path / f"nested-acl-after-launchers-{target_role}"
    copy_inputs(root)
    original = checkpoint._invoke_launcher
    observed_roots: dict[str, Path] = {}

    def invoke_then_add_acl(*args: object, **kwargs: object) -> dict:
        result = original(*args, **kwargs)
        operation = args[3]
        assert operation in {"ORACLE", "VERIFIER"}
        prefix = str(operation).lower()
        observed_roots[f"{prefix}_input_root"] = Path(args[1])
        observed_roots[f"{prefix}_output_root"] = Path(args[2])
        if operation == "VERIFIER":
            nested_name = (
                "inputs"
                if target_role.endswith("input_root")
                else "oracle_output"
                if target_role == "oracle_output_root"
                else "verifier_output"
            )
            _add_test_acl(observed_roots[target_role] / nested_name)
        return result

    monkeypatch.setattr(checkpoint, "_invoke_launcher", invoke_then_add_acl)
    with pytest.raises(checkpoint.EvidenceError, match="extended ACL"):
        checkpoint.compute(root)


def test_runtime_scratch_integrity_allows_nested_directory_xattrs_after_launchers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "nested-xattr-after-launchers"
    copy_inputs(root)
    original = checkpoint._invoke_launcher
    observed_roots: dict[str, Path] = {}

    def invoke_then_add_xattrs(*args: object, **kwargs: object) -> dict:
        result = original(*args, **kwargs)
        operation = args[3]
        assert operation in {"ORACLE", "VERIFIER"}
        prefix = str(operation).lower()
        observed_roots[f"{prefix}_input_root"] = Path(args[1])
        observed_roots[f"{prefix}_output_root"] = Path(args[2])
        if operation == "VERIFIER":
            for target_role, nested_name in (
                ("oracle_input_root", "inputs"),
                ("oracle_output_root", "oracle_output"),
                ("verifier_input_root", "inputs"),
                ("verifier_output_root", "verifier_output"),
            ):
                _add_test_xattr(observed_roots[target_role] / nested_name)
        return result

    monkeypatch.setattr(checkpoint, "_invoke_launcher", invoke_then_add_xattrs)
    artifacts, audit = checkpoint.compute(root)
    assert set(artifacts) == set(checkpoint.TOTAL_ARTIFACT_FILES)
    assert audit["classification"] == checkpoint.RUNTIME_CLASSIFICATION


def test_partial_publication_has_no_terminal_commit_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "partial-recovery"
    copy_inputs(root)
    original = checkpoint.atomic_bytes_at
    failed = False

    def fail_before_one_runtime_artifact(
        root_fd: int, relative: str, value: bytes
    ) -> None:
        nonlocal failed
        if not failed and relative.endswith("oracle_manifest.json"):
            failed = True
            raise OSError("injected checkpoint publication fault")
        original(root_fd, relative, value)

    monkeypatch.setattr(checkpoint, "atomic_bytes_at", fail_before_one_runtime_artifact)
    with pytest.raises(OSError, match="injected checkpoint publication fault"):
        checkpoint.build(root)
    assert not (root / checkpoint.AUDIT_PATH).exists()
    assert not (root / checkpoint.CHECKPOINT_COMMIT_PATH).exists()

    monkeypatch.setattr(checkpoint, "atomic_bytes_at", original)
    payload = checkpoint.build(root)
    assert_boundary(payload)
    assert (root / checkpoint.CHECKPOINT_COMMIT_PATH).is_file()


def test_preexisting_terminal_commit_is_invalidated_when_rebuild_compute_fails(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stale-terminal"
    build_copy(root)
    terminal = root / checkpoint.CHECKPOINT_COMMIT_PATH
    assert terminal.is_file()
    result_relative = checkpoint.LEGACY_RUN_ARTIFACTS[25][0]
    result = root / result_relative
    result.write_bytes(result.read_bytes() + b"\n")

    rejected = run_builder(root, "build")
    assert rejected.returncode != 0
    assert "legacy frozen artifact set hash mismatch" in rejected.stderr
    assert not terminal.exists()


def test_fault_after_terminal_link_invalidates_exact_terminal_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "terminal-link-fault"
    copy_inputs(root)
    original = checkpoint.atomic_bytes_at

    def publish_then_fail(root_fd: int, relative: str, value: bytes) -> None:
        original(root_fd, relative, value)
        if relative == checkpoint.CHECKPOINT_COMMIT_PATH:
            raise OSError("fault after terminal link")

    monkeypatch.setattr(checkpoint, "atomic_bytes_at", publish_then_fail)
    with pytest.raises(OSError, match="fault after terminal link"):
        checkpoint.build(root)
    assert not (root / checkpoint.CHECKPOINT_COMMIT_PATH).exists()


def test_lock_replacement_inside_terminal_publication_invalidates_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "terminal-lock-replacement"
    copy_inputs(root)
    original = checkpoint.atomic_bytes_at

    def publish_then_replace_lock(root_fd: int, relative: str, value: bytes) -> None:
        original(root_fd, relative, value)
        if relative == checkpoint.CHECKPOINT_COMMIT_PATH:
            lock = root / checkpoint.BUILDER_LOCK_PATH
            lock_bytes = lock.read_bytes()
            lock.unlink()
            lock.write_bytes(lock_bytes)
            lock.chmod(0o644)

    monkeypatch.setattr(checkpoint, "atomic_bytes_at", publish_then_replace_lock)
    with pytest.raises(
        checkpoint.EvidenceError, match="builder lock pathname identity changed"
    ):
        checkpoint.build(root)
    assert not (root / checkpoint.CHECKPOINT_COMMIT_PATH).exists()


def test_final_evidence_byte_mutation_invalidates_terminal_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "final-evidence-mutation"
    copy_inputs(root)
    original = checkpoint._evidence_file_set_at
    calls = 0

    def inspect_then_mutate(root_fd: int, expected: set[str]) -> set[str]:
        nonlocal calls
        calls += 1
        result = original(root_fd, expected)
        if calls == 2:
            artifact = root / checkpoint.EVIDENCE_ROOT / "inputs/source_manifest.json"
            artifact.write_bytes(artifact.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(checkpoint, "_evidence_file_set_at", inspect_then_mutate)
    with pytest.raises(
        checkpoint.EvidenceError,
        match="checkpoint evidence bytes changed before return",
    ):
        checkpoint.build(root)
    assert calls == 2
    assert not (root / checkpoint.CHECKPOINT_COMMIT_PATH).exists()


def test_kill_after_final_link_before_temp_unlink_recovers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "linked-partial-recovery"
    build_copy(root)
    final = root / checkpoint.EVIDENCE_ROOT / "inputs/source_manifest.json"
    partial = final.parent / f".{final.name}.builder-partial-deadprocess"
    os.link(final, partial)
    assert final.stat().st_nlink == 2

    recovered = run_builder(root, "build")
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    assert not partial.exists()
    assert final.stat().st_nlink == 1
    assert (root / checkpoint.CHECKPOINT_COMMIT_PATH).is_file()


def test_kill_after_terminal_link_before_temp_unlink_recovers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "terminal-linked-partial-recovery"
    build_copy(root)
    terminal = root / checkpoint.CHECKPOINT_COMMIT_PATH
    partial = terminal.parent / f".{terminal.name}.builder-partial-deadprocess"
    os.link(terminal, partial)
    assert terminal.stat().st_nlink == 2

    recovered = run_builder(root, "build")
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    assert not partial.exists()
    assert terminal.stat().st_nlink == 1
    assert run_builder(root, "validate").returncode == 0


def test_external_hardlink_alias_invalidates_checkpoint_validation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "hardlink-alias"
    build_copy(root)
    evidence = root / checkpoint.EVIDENCE_ROOT / "inputs/source_manifest.json"
    outside_alias = tmp_path / "outside-source-manifest-alias.json"
    os.link(evidence, outside_alias)
    rejected = run_builder(root, "validate")
    assert rejected.returncode != 0
    assert "not an owner-controlled regular file" in rejected.stderr
    outside_alias.unlink()
    accepted = run_builder(root, "validate")
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr


def test_concurrent_builders_have_one_lock_owner_and_one_terminal_commit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "concurrent-builders"
    copy_inputs(root)
    arguments = [
        str(PYTHON),
        "-I",
        "-S",
        "-B",
        str(root / checkpoint.BUILDER_PATH),
        "build",
        "--root",
        str(root),
    ]
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    first = subprocess.Popen(
        arguments,
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    second = subprocess.Popen(
        arguments,
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    first_output = first.communicate(timeout=60)
    second_output = second.communicate(timeout=60)
    outcomes = [(first.returncode, *first_output), (second.returncode, *second_output)]
    assert sorted(item[0] for item in outcomes) == [0, 1]
    failure = next(item for item in outcomes if item[0] != 0)
    assert "another checkpoint builder owns the lock" in failure[2]
    lock_path = root / checkpoint.BUILDER_LOCK_PATH
    assert lock_path.read_bytes() == b"PAPER_RESEARCH_JPY_ORACLE_CORRECTIVE_V2_LOCK_V1\n"
    validation = run_builder(root, "validate")
    assert validation.returncode == 0, validation.stdout + validation.stderr
    assert (root / checkpoint.CHECKPOINT_COMMIT_PATH).is_file()


def test_root_swap_after_compute_fails_without_redirected_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root-swap"
    copy_inputs(root)
    moved = tmp_path / "root-swap-original"
    outside = tmp_path / "root-swap-outside"
    outside.mkdir()
    original_compute = checkpoint.compute

    def compute_then_swap(supplied_root: Path) -> tuple[dict[str, bytes], dict]:
        result = original_compute(supplied_root)
        os.rename(root, moved)
        root.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(checkpoint, "compute", compute_then_swap)
    with pytest.raises(checkpoint.EvidenceError, match="root path identity changed"):
        checkpoint.build(root)
    assert not (outside / checkpoint.EVIDENCE_ROOT).exists()
    assert not (moved / checkpoint.EVIDENCE_ROOT).exists()


def test_launcher_change_after_source_freeze_fails_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "launcher-race"
    copy_inputs(root)
    original_invoke = checkpoint._invoke_launcher
    mutated = False

    def mutate_then_invoke(*args: object, **kwargs: object) -> dict:
        nonlocal mutated
        if not mutated:
            launcher_path = root / checkpoint.LAUNCHER_PATH
            launcher_path.write_bytes(launcher_path.read_bytes() + b"\n# race mutation\n")
            mutated = True
        return original_invoke(*args, **kwargs)

    monkeypatch.setattr(checkpoint, "_invoke_launcher", mutate_then_invoke)
    with pytest.raises(checkpoint.EvidenceError, match="changed after source freeze"):
        checkpoint.compute(root)
    assert not (root / checkpoint.EVIDENCE_ROOT).exists()


def test_nonruntime_source_change_after_execution_fails_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "nonruntime-source-race"
    copy_inputs(root)
    original_invoke = checkpoint._invoke_launcher
    calls = 0

    def invoke_then_mutate(*args: object, **kwargs: object) -> dict:
        nonlocal calls
        result = original_invoke(*args, **kwargs)
        calls += 1
        if calls == 2:
            source = root / checkpoint.ORACLE_TEST_PATH
            source.write_bytes(source.read_bytes() + b"\n# late source mutation\n")
        return result

    monkeypatch.setattr(checkpoint, "_invoke_launcher", invoke_then_mutate)
    with pytest.raises(
        checkpoint.EvidenceError,
        match="checkpoint source files changed after freeze",
    ):
        checkpoint.compute(root)
    assert calls == 2
    assert not (root / checkpoint.EVIDENCE_ROOT).exists()


def test_legacy_v1_change_after_execution_fails_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "legacy-v1-race"
    copy_inputs(root)
    original_invoke = checkpoint._invoke_launcher
    calls = 0

    def invoke_then_mutate(*args: object, **kwargs: object) -> dict:
        nonlocal calls
        result = original_invoke(*args, **kwargs)
        calls += 1
        if calls == 2:
            source = root / "evidence/paper_research_jpy_oracle_v1/oracle_ledger_v1.jsonl"
            source.write_bytes(source.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(checkpoint, "_invoke_launcher", invoke_then_mutate)
    with pytest.raises(
        checkpoint.EvidenceError,
        match="legacy V1 aggregate differs from its fixed review hash",
    ):
        checkpoint.compute(root)
    assert calls == 2
    assert not (root / checkpoint.EVIDENCE_ROOT).exists()


def test_legacy_cycle_change_after_execution_fails_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "legacy-cycle-race"
    copy_inputs(root)
    original_invoke = checkpoint._invoke_launcher
    calls = 0

    def invoke_then_mutate(*args: object, **kwargs: object) -> dict:
        nonlocal calls
        result = original_invoke(*args, **kwargs)
        calls += 1
        if calls == 2:
            source = root / "evidence/orchestrator_state_v2/official_seal_v25.json"
            source.write_bytes(source.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(checkpoint, "_invoke_launcher", invoke_then_mutate)
    with pytest.raises(
        checkpoint.EvidenceError,
        match="legacy frozen artifact set hash mismatch",
    ):
        checkpoint.compute(root)
    assert calls == 2
    assert not (root / checkpoint.EVIDENCE_ROOT).exists()


def test_source_change_during_publication_prevents_terminal_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "publication-source-race"
    copy_inputs(root)
    original_atomic = checkpoint.atomic_bytes_at
    mutated = False

    def publish_then_mutate(root_fd: int, relative: str, value: bytes) -> None:
        nonlocal mutated
        original_atomic(root_fd, relative, value)
        if not mutated:
            source = root / checkpoint.ORACLE_TEST_PATH
            source.write_bytes(source.read_bytes() + b"\n# publication mutation\n")
            mutated = True

    monkeypatch.setattr(checkpoint, "atomic_bytes_at", publish_then_mutate)
    with pytest.raises(
        checkpoint.EvidenceError,
        match="checkpoint source files changed before terminal commit",
    ):
        checkpoint.build(root)
    assert mutated is True
    assert not (root / checkpoint.CHECKPOINT_COMMIT_PATH).exists()


@pytest.mark.parametrize(
    ("target_relative", "expected_error"),
    (
        (
            checkpoint.ORACLE_TEST_PATH,
            "checkpoint source files changed after terminal commit",
        ),
        (
            "evidence/orchestrator_state_v2/official_seal_v25.json",
            "legacy frozen artifact set hash mismatch",
        ),
        (
            checkpoint.LEGACY_RUN_ARTIFACTS[41][1],
            "legacy frozen artifact set hash mismatch",
        ),
        (
            "evidence/paper_research_jpy_oracle_v1/oracle_ledger_v1.jsonl",
            "legacy V1 aggregate differs from its fixed review hash",
        ),
    ),
)
def test_binding_change_inside_terminal_publication_invalidates_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_relative: str,
    expected_error: str,
) -> None:
    root = tmp_path / (Path(target_relative).stem + "-terminal-race")
    copy_inputs(root)
    original_atomic = checkpoint.atomic_bytes_at
    mutated = False

    def mutate_inside_terminal_link(root_fd: int, relative: str, value: bytes) -> None:
        nonlocal mutated
        if relative == checkpoint.CHECKPOINT_COMMIT_PATH and not mutated:
            target = root / target_relative
            target.write_bytes(target.read_bytes() + b"\n")
            mutated = True
        original_atomic(root_fd, relative, value)

    monkeypatch.setattr(checkpoint, "atomic_bytes_at", mutate_inside_terminal_link)
    with pytest.raises(checkpoint.EvidenceError, match=expected_error):
        checkpoint.build(root)
    assert mutated is True
    assert not (root / checkpoint.CHECKPOINT_COMMIT_PATH).exists()


def test_builder_lock_unlink_recreate_prevents_terminal_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "builder-lock-replacement"
    copy_inputs(root)
    original_compute = checkpoint.compute

    def compute_then_replace_lock(supplied_root: Path) -> tuple[dict[str, bytes], dict]:
        result = original_compute(supplied_root)
        lock = root / checkpoint.BUILDER_LOCK_PATH
        value = lock.read_bytes()
        lock.unlink()
        lock.write_bytes(value)
        lock.chmod(0o644)
        return result

    monkeypatch.setattr(checkpoint, "compute", compute_then_replace_lock)
    with pytest.raises(
        checkpoint.EvidenceError,
        match="builder lock pathname identity changed",
    ):
        checkpoint.build(root)
    assert not (root / checkpoint.CHECKPOINT_COMMIT_PATH).exists()


@pytest.mark.parametrize(
    "relative",
    (
        checkpoint.LAUNCHER_PATH,
        checkpoint.GOLDEN_PATH,
        checkpoint.REFERENCE_PATH,
        checkpoint.REFERENCE_CONTRACT_PATH,
        checkpoint.REFERENCE_TEST_PATH,
        checkpoint.REFERENCE_MUTATION_TEST_PATH,
    ),
)
def test_runtime_golden_or_reference_source_tamper_invalidates_existing_checkpoint(
    tmp_path: Path,
    relative: str,
) -> None:
    root = tmp_path / Path(relative).stem
    build_copy(root)
    path = root / relative
    path.write_bytes(path.read_bytes() + b"\n# tamper\n")
    completed = run_builder(root, "validate")
    assert completed.returncode != 0


def test_symlinked_runtime_or_evidence_parent_fails_closed(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime-symlink"
    copy_inputs(runtime_root)
    launcher = runtime_root / checkpoint.LAUNCHER_PATH
    launcher.unlink()
    launcher.symlink_to(ROOT / checkpoint.LAUNCHER_PATH)
    runtime_failure = run_builder(runtime_root, "build")
    assert runtime_failure.returncode != 0
    assert not (runtime_root / checkpoint.EVIDENCE_ROOT).exists()

    evidence_root = tmp_path / "evidence-symlink"
    copy_inputs(evidence_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (evidence_root / checkpoint.EVIDENCE_ROOT).symlink_to(
        outside, target_is_directory=True
    )
    evidence_failure = run_builder(evidence_root, "build")
    assert evidence_failure.returncode != 0
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "forged"),
    (
        ("paper_only", 1),
        ("live_authority", 0),
        ("external_orders", False),
        ("external_orders", 0.0),
    ),
)
def test_authority_boundary_rejects_bool_integer_aliases(
    field: str, forged: object
) -> None:
    authority = dict(checkpoint.PAPER_ONLY_AUTHORITY)
    authority[field] = forged
    with pytest.raises(checkpoint.EvidenceError, match="exact (boolean|integer)"):
        checkpoint._validate_exact_authority(authority, "forged")


def test_v1_binding_requires_fixed_git_commit_tree_subtree_and_blobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "v1-git-binding"
    copy_inputs(root)
    binding = checkpoint._legacy_v1_evidence_binding(root)
    assert binding["aggregate_sha256"] == checkpoint.SUPERSEDED_V1_AGGREGATE_SHA256
    assert binding["prior_embedded_audit_sha256"] == (
        checkpoint.SUPERSEDED_V1_AUDIT_SHA256
    )
    assert binding["audit_file_sha256"] == checkpoint.SUPERSEDED_V1_AUDIT_FILE_SHA256
    assert binding["git_binding"]["commit"] == checkpoint.SUPERSEDED_COMMIT
    assert binding["git_binding"]["commit_tree"] == checkpoint.SUPERSEDED_COMMIT_TREE
    assert binding["git_binding"]["subtree"] == checkpoint.SUPERSEDED_V1_SUBTREE
    monkeypatch.setattr(checkpoint, "SUPERSEDED_V1_SUBTREE", "0" * 40)
    with pytest.raises(checkpoint.EvidenceError, match="commit/tree/subtree"):
        checkpoint._legacy_v1_evidence_binding(root)


def test_joint_v1_reseal_cannot_replace_fixed_bundle(tmp_path: Path) -> None:
    root = tmp_path / "v1-joint-reseal"
    copy_inputs(root)
    v1 = root / "evidence/paper_research_jpy_oracle_v1"
    ledger = v1 / "oracle_ledger_v1.jsonl"
    ledger.write_bytes(ledger.read_bytes() + b"\n")
    audit_path = v1 / "oracle_checkpoint_v1.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    relative = "evidence/paper_research_jpy_oracle_v1/oracle_ledger_v1.jsonl"
    audit["evidence_artifact_sha256"][relative] = checkpoint.sha256_file(ledger)
    audit["audit_sha256"] = checkpoint.embedded(audit, "audit_sha256")
    audit_path.write_bytes(checkpoint.canonical(audit) + b"\n")
    with pytest.raises(
        checkpoint.EvidenceError, match="differs from (?:its )?fixed review"
    ):
        checkpoint._legacy_v1_evidence_binding(root)


def test_v1_working_mode_must_match_fixed_git_tree(tmp_path: Path) -> None:
    root = tmp_path / "v1-mode"
    copy_inputs(root)
    target = root / "evidence/paper_research_jpy_oracle_v1/oracle_ledger_v1.jsonl"
    target.chmod(0o755)
    with pytest.raises(checkpoint.EvidenceError, match="mode/type changed"):
        checkpoint._legacy_v1_evidence_binding(root)


@pytest.mark.parametrize(
    "tamper",
    (
        "oracle_commit", "oracle_authority_type", "verifier_admission",
        "unknown_field", "arm_metric_unknown", "arm_metric_bool_alias",
        "reference_descriptor_path", "reference_descriptor_sha",
        "reference_descriptor_size", "reference_receipt_code_pin",
        "reference_receipt_contract_pin", "reference_receipt_input_root",
        "reference_receipt_projection", "reference_receipt_journal_count",
        "reference_receipt_journal_count_type", "reference_receipt_balanced",
        "reference_receipt_balanced_type", "reference_receipt_diagnostics",
        "reference_receipt_n_eff", "reference_receipt_direction_gate",
        "reference_receipt_result_hash", "reference_receipt_result_hash_type",
        "reference_content_binding_code_pin",
        "reference_content_binding_result_hash", "reference_receipt_unknown",
    ),
)
def test_inner_output_chain_and_semantic_boundary_fail_closed(
    tmp_path: Path, tamper: str
) -> None:
    root = tmp_path / f"inner-{tamper}"
    payload = build_copy(root)
    state = root / checkpoint.EVIDENCE_ROOT
    oracle_launch = json.loads(
        (root / checkpoint.ORACLE_LAUNCH_RECEIPT_PATH).read_text(encoding="utf-8")
    )
    verifier_launch = json.loads(
        (root / checkpoint.VERIFIER_LAUNCH_RECEIPT_PATH).read_text(encoding="utf-8")
    )
    if tamper == "oracle_commit":
        path = state / "oracle_output/COMMIT.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["request_sha256"] = "0" * 64
        path.write_bytes(checkpoint.canonical(value) + b"\n")
    elif tamper in {
        "oracle_authority_type", "unknown_field", "arm_metric_unknown",
        "arm_metric_bool_alias",
    }:
        manifest_path = state / "oracle_output/oracle_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if tamper == "oracle_authority_type":
            manifest["authority"]["paper_only"] = 1
        elif tamper == "unknown_field":
            manifest["profit_gate_pass"] = True
        else:
            raw_metrics = manifest["oracle_metrics"]["arms"]["RAW_SIGNAL"]
            if tamper == "arm_metric_unknown":
                raw_metrics["profit_gate_pass"] = False
            else:
                raw_metrics["proposal_count"] = True
            manifest["oracle_metrics"]["metrics_sha256"] = checkpoint.embedded(
                manifest["oracle_metrics"], "metrics_sha256"
            )
        manifest["oracle_root_sha256"] = checkpoint.embedded(
            manifest, "oracle_root_sha256"
        )
        manifest_bytes = checkpoint.canonical(manifest) + b"\n"
        manifest_path.write_bytes(manifest_bytes)
        commit_path = state / "oracle_output/COMMIT.json"
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        commit["manifest_sha256"] = checkpoint.sha256_bytes(manifest_bytes)
        commit["manifest_size_bytes"] = len(manifest_bytes)
        commit_path.write_bytes(checkpoint.canonical(commit) + b"\n")
    elif tamper.startswith("reference_descriptor_"):
        request_path = state / "inputs/verifier_request.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        descriptor = request["reference_code_snapshot"]
        if tamper == "reference_descriptor_path":
            descriptor["relative_path"] = "inputs/reference_contract_snapshot.json"
        elif tamper == "reference_descriptor_sha":
            descriptor["sha256"] = "0" * 64
        else:
            descriptor["size_bytes"] += 1
        request_path.write_bytes(checkpoint.canonical(request) + b"\n")
    else:
        receipt_path = state / "verifier_output/verifier_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if tamper == "verifier_admission":
            receipt["admission_eligible"] = True
        elif tamper == "reference_receipt_code_pin":
            receipt["reference_code_sha256"] = "0" * 64
        elif tamper == "reference_receipt_contract_pin":
            receipt["reference_contract_sha256"] = "0" * 64
        elif tamper == "reference_receipt_input_root":
            receipt["reference_input_root_sha256"] = "0" * 64
        elif tamper == "reference_receipt_projection":
            receipt["reference_economic_projection_sha256"] = "0" * 64
        elif tamper == "reference_receipt_journal_count":
            receipt["reference_journal_transaction_count"] = 23
        elif tamper == "reference_receipt_journal_count_type":
            receipt["reference_journal_transaction_count"] = True
        elif tamper == "reference_receipt_balanced":
            receipt["reference_all_transactions_balanced"] = False
        elif tamper == "reference_receipt_balanced_type":
            receipt["reference_all_transactions_balanced"] = 1
        elif tamper == "reference_receipt_diagnostics":
            receipt["reference_accounting_diagnostics_only"] = False
        elif tamper == "reference_receipt_n_eff":
            receipt["reference_n_eff_statistical_admission_allowed"] = True
        elif tamper == "reference_receipt_direction_gate":
            receipt["reference_direction_accuracy_profit_gate_allowed"] = True
        elif tamper == "reference_receipt_result_hash":
            receipt["reference_result_sha256"] = "0" * 64
        elif tamper == "reference_receipt_result_hash_type":
            receipt["reference_result_sha256"] = False
        elif tamper == "reference_content_binding_code_pin":
            receipt["verifier_release_content_binding"][
                "reference_code_sha256"
            ] = "0" * 64
        elif tamper == "reference_content_binding_result_hash":
            receipt["verifier_release_content_binding"][
                "reference_result_sha256"
            ] = "0" * 64
        else:
            receipt["reference_profit_gate_pass"] = True
        receipt["verifier_receipt_sha256"] = checkpoint.embedded(
            receipt, "verifier_receipt_sha256"
        )
        receipt_bytes = checkpoint.canonical(receipt) + b"\n"
        receipt_path.write_bytes(receipt_bytes)
        commit_path = state / "verifier_output/COMMIT.json"
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        commit["receipt_sha256"] = checkpoint.sha256_bytes(receipt_bytes)
        commit["receipt_size_bytes"] = len(receipt_bytes)
        commit["verifier_receipt_sha256"] = receipt["verifier_receipt_sha256"]
        commit_path.write_bytes(checkpoint.canonical(commit) + b"\n")
    with pytest.raises(checkpoint.EvidenceError):
        checkpoint._validate_inner_runtime_outputs(
            state,
            state,
            state,
            state,
            payload["source_artifact_sha256"],
            oracle_launch,
            verifier_launch,
        )


@pytest.fixture(scope="module")
def generated_current_lifecycle(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, bytes], dict]:
    current_root = tmp_path_factory.mktemp("current-lifecycle") / "root"
    audit = build_copy(current_root)
    validation = run_builder(current_root, "validate")
    assert validation.returncode == 0, validation.stdout + validation.stderr
    artifacts = {
        relative: (current_root / relative).read_bytes()
        for relative in checkpoint.TOTAL_ARTIFACT_FILES
    }
    return artifacts, audit


def test_evidence_lifecycle_policy_derives_legacy_and_validated_current_state(
) -> None:
    contract = json.loads((ROOT / checkpoint.CONTRACT_PATH).read_text(encoding="utf-8"))
    assert contract["evidence_lifecycle_policy"] \
        == EXPECTED_EVIDENCE_LIFECYCLE_POLICY
    assert "checked_in_evidence_state" not in contract
    assert "regeneration_performed_by_this_change" not in contract

    source_hashes = _source_hashes_only(ROOT)
    assert _derive_evidence_lifecycle_state(
        *_literal_legacy_lifecycle_fixture(),
        current_validation_succeeded=False,
        expected_source_hashes=source_hashes,
    ) == LEGACY_LIFECYCLE_STATE

    current_chain = _synthetic_current_lifecycle_fixture(source_hashes)
    assert _derive_evidence_lifecycle_state(
        *current_chain,
        current_validation_succeeded=True,
        expected_source_hashes=source_hashes,
    ) == CURRENT_LIFECYCLE_STATE


def test_compute_generates_exact_current_28_artifact_lifecycle(
    generated_current_lifecycle: tuple[dict[str, bytes], dict],
) -> None:
    artifacts, audit = generated_current_lifecycle
    assert len(artifacts) == 28
    assert set(artifacts) == set(checkpoint.TOTAL_ARTIFACT_FILES)
    assert audit == json.loads(artifacts[checkpoint.AUDIT_PATH])
    receipt = json.loads(artifacts[REFERENCE_RECEIPT_PATH])
    terminal = json.loads(artifacts[checkpoint.CHECKPOINT_COMMIT_PATH])
    assert receipt["classification"] == CURRENT_LIFECYCLE_STATE
    assert receipt["reference_result_sha256"] == receipt[
        "verifier_release_content_binding"
    ]["reference_result_sha256"]
    assert terminal["artifact_count"] == 27
    assert len(terminal["artifact_sha256"]) == 27
    assert _derive_evidence_lifecycle_state(
        artifacts[checkpoint.AUDIT_PATH],
        artifacts[REFERENCE_RECEIPT_PATH],
        artifacts[checkpoint.CHECKPOINT_COMMIT_PATH],
        current_validation_succeeded=True,
        expected_source_hashes=_source_hashes_only(ROOT),
    ) == CURRENT_LIFECYCLE_STATE


@pytest.mark.parametrize(
    "tamper",
    (
        "receipt_reference_absent",
        "receipt_reference_partial",
        "binding_reference_partial",
        "reference_mismatch",
        "receipt_release_true",
        "audit_admission_true",
        "terminal_admission_true",
        "source_set_incomplete",
        "validation_failed",
        "validation_bool_coercion",
    ),
)
def test_current_lifecycle_rejects_mixed_partial_or_nonvalidated_state(
    tamper: str,
) -> None:
    source_hashes = _source_hashes_only(ROOT)
    audit_bytes, receipt_bytes, terminal_bytes = (
        _synthetic_current_lifecycle_fixture(source_hashes)
    )
    audit = json.loads(audit_bytes)
    receipt = json.loads(receipt_bytes)
    terminal = json.loads(terminal_bytes)
    validation_succeeded: object = True
    if tamper == "receipt_reference_absent":
        receipt.pop("reference_result_sha256")
        receipt["verifier_release_content_binding"].pop("reference_result_sha256")
    elif tamper == "receipt_reference_partial":
        receipt.pop("reference_result_sha256")
    elif tamper == "binding_reference_partial":
        receipt["verifier_release_content_binding"].pop("reference_result_sha256")
    elif tamper == "reference_mismatch":
        receipt["verifier_release_content_binding"]["reference_result_sha256"] \
            = "0" * 64
    elif tamper == "receipt_release_true":
        receipt["release_evidence_eligible"] = True
    elif tamper == "audit_admission_true":
        audit["strategy_admission_eligible"] = True
    elif tamper == "terminal_admission_true":
        terminal["strategy_admission_eligible"] = True
    elif tamper == "source_set_incomplete":
        audit["source_artifact_sha256"].pop(checkpoint.CONTRACT_PATH)
    elif tamper == "validation_failed":
        validation_succeeded = False
    else:
        validation_succeeded = 1
    audit_bytes, receipt_bytes, terminal_bytes = _reseal_lifecycle_chain(
        audit, receipt, terminal
    )
    with pytest.raises(ValueError):
        _derive_evidence_lifecycle_state(
            audit_bytes,
            receipt_bytes,
            terminal_bytes,
            current_validation_succeeded=validation_succeeded,  # type: ignore[arg-type]
            expected_source_hashes=source_hashes,
        )


@pytest.mark.parametrize(
    "tamper",
    ("partial_reference", "receipt_release_true", "validation_succeeded"),
)
def test_legacy_lifecycle_rejects_attesting_partial_or_current_claim(
    tamper: str,
) -> None:
    audit_bytes, receipt_bytes, terminal_bytes = _literal_legacy_lifecycle_fixture()
    audit = json.loads(audit_bytes)
    receipt = json.loads(receipt_bytes)
    terminal = json.loads(terminal_bytes)
    validation_succeeded = False
    if tamper == "partial_reference":
        receipt["reference_result_sha256"] = "0" * 64
    elif tamper == "receipt_release_true":
        receipt["release_evidence_eligible"] = True
    else:
        validation_succeeded = True
    audit_bytes, receipt_bytes, terminal_bytes = _reseal_lifecycle_chain(
        audit, receipt, terminal
    )
    with pytest.raises(ValueError):
        _derive_evidence_lifecycle_state(
            audit_bytes,
            receipt_bytes,
            terminal_bytes,
            current_validation_succeeded=validation_succeeded,
            expected_source_hashes=_source_hashes_only(ROOT),
        )


def test_lifecycle_source_suite_is_hermetic_to_repository_evidence_state(
    tmp_path: Path,
) -> None:
    assert not any(
        relative == checkpoint.EVIDENCE_ROOT
        or relative.startswith(f"{checkpoint.EVIDENCE_ROOT}/")
        for relative in checkpoint.SOURCE_FILES
    )
    current_chain = _synthetic_current_lifecycle_fixture(
        _source_hashes_only(ROOT)
    )
    legacy_chain = _literal_legacy_lifecycle_fixture()
    observations = []
    for state in ("source-only", "legacy-stale", "current-generated"):
        root = tmp_path / state
        for relative in checkpoint.SOURCE_FILES:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        if state != "source-only":
            chain = legacy_chain if state == "legacy-stale" else current_chain
            for relative, raw in zip(
                (
                    checkpoint.AUDIT_PATH,
                    REFERENCE_RECEIPT_PATH,
                    checkpoint.CHECKPOINT_COMMIT_PATH,
                ),
                chain,
                strict=True,
            ):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
        observations.append(_source_only_lifecycle_observation(root))
    assert observations[0] == observations[1] == observations[2]
