from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import paper_research_fd_launcher_v2 as launcher
from test_paper_research_oracle_verifier_v2 import artifact, build_inputs, write_json


ROOT = Path(__file__).resolve().parent
PYTHON = Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3")
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def add_test_acl(path: Path, *, inherited: bool = False) -> None:
    rule = "everyone allow read"
    if inherited:
        rule += ",file_inherit,directory_inherit"
    subprocess.run(("/bin/chmod", "+a", rule, str(path)), check=True)


def clear_test_acl(path: Path) -> None:
    subprocess.run(("/bin/chmod", "-N", str(path)), check=True)


def add_test_xattr(path: Path) -> None:
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


def pure_verifier_source(body: bytes = b"return (b'{}\\n', b'{}\\n')") -> bytes:
    return (
        b"_SEALED_RUNTIME = True\n"
        b"def verify_sealed_bytes(request_bytes, artifact_blobs, "
        b"oracle_release_blobs, reference_result_bytes, "
        b"reference_attestation):\n    "
        + body.replace(b"\n", b"\n    ")
        + b"\n"
    )


def runtime_copies(root: Path) -> dict[str, object]:
    runtime = root / "runtime"
    runtime.mkdir(parents=True, mode=0o700)
    sources = {
        "launcher": ROOT / "paper_research_fd_launcher_v2.py",
        "oracle_code": ROOT / "paper_research_jpy_oracle_v2.py",
        "oracle_contract": ROOT / "PAPER_RESEARCH_JPY_ORACLE_CONTRACT_V2.json",
        "oracle_schema": ROOT / "paper_research_jpy_oracle_schema_v2.json",
        "verifier_code": ROOT / "paper_research_oracle_verifier_v2.py",
        "verifier_schema": ROOT / "paper_research_oracle_verifier_schema_v2.json",
        "reference_code": ROOT / "paper_research_double_entry_reference_v2.py",
        "reference_contract": ROOT
        / "PAPER_RESEARCH_DOUBLE_ENTRY_REFERENCE_CONTRACT_V2.json",
    }
    result: dict[str, object] = {}
    for label, source in sources.items():
        data = source.read_bytes()
        target = runtime / source.name
        target.write_bytes(data)
        result[label] = target
        result[f"{label}_bytes"] = data
    return result


def _open_readonly(path: Path) -> int:
    return os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))


def assert_launcher_failure(
    completed: subprocess.CompletedProcess[str],
    message: str,
) -> None:
    assert completed.returncode == 2
    assert json.loads(completed.stdout) == {
        "error_code": "SEALED_FD_LAUNCHER_FAIL_CLOSED",
        "error_sha256": digest(message.encode("utf-8")),
        "ok": False,
    }


def launch_output_root(state_root: Path, operation: str) -> Path:
    return state_root / f"{operation.lower()}-publish"


def invoke_launcher(
    runtime: dict[str, object],
    *,
    operation: str,
    request_path: Path,
    state_root: Path,
    output_root: Path | None = None,
    mutate_after_open: callable | None = None,
    overrides: dict[str, Path] | None = None,
    python_flags: tuple[str, ...] = ("-I", "-S", "-B"),
    bootstrap_source: str | None = None,
    expected_launcher_sha256: str | None = None,
    omit_roles: frozenset[str] = frozenset(),
    oracle_reference_fds: bool = False,
) -> subprocess.CompletedProcess[str]:
    overrides = {} if overrides is None else overrides
    if output_root is None:
        output_root = launch_output_root(state_root, operation)
        output_root.mkdir(mode=0o700)
    bootstrap_source = (
        launcher.FIXED_BOOTSTRAP_SOURCE
        if bootstrap_source is None
        else bootstrap_source
    )
    expected_launcher_sha256 = (
        digest(runtime["launcher_bytes"])
        if expected_launcher_sha256 is None
        else expected_launcher_sha256
    )
    code_role = "oracle_code" if operation == "ORACLE" else "verifier_code"
    schema_role = "oracle_schema" if operation == "ORACLE" else "verifier_schema"
    paths = {
        "request": request_path,
        "code": overrides.get("code", runtime[code_role]),
        "schema": overrides.get("schema", runtime[schema_role]),
    }
    if operation == "ORACLE":
        paths["contract"] = overrides.get("contract", runtime["oracle_contract"])
        if oracle_reference_fds:
            paths.update({
                "reference_code": overrides.get(
                    "reference_code", runtime["reference_code"]
                ),
                "reference_contract": overrides.get(
                    "reference_contract", runtime["reference_contract"]
                ),
            })
    else:
        paths.update({
            "oracle_code": overrides.get("oracle_code", runtime["oracle_code"]),
            "oracle_contract": overrides.get("oracle_contract", runtime["oracle_contract"]),
            "oracle_schema": overrides.get("oracle_schema", runtime["oracle_schema"]),
            "reference_code": overrides.get("reference_code", runtime["reference_code"]),
            "reference_contract": overrides.get(
                "reference_contract", runtime["reference_contract"]
            ),
        })
    descriptors: dict[str, int] = {
        "launcher": _open_readonly(runtime["launcher"]),
        "request": _open_readonly(paths["request"]),
        "input_root": os.open(state_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)),
        "output_root": os.open(output_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)),
        "code": _open_readonly(paths["code"]),
        "schema": _open_readonly(paths["schema"]),
    }
    for role in (
        "contract",
        "oracle_code",
        "oracle_contract",
        "oracle_schema",
        "reference_code",
        "reference_contract",
    ):
        if role in paths and role not in omit_roles:
            descriptors[role] = _open_readonly(paths[role])
    arguments = [
        str(PYTHON),
        *python_flags,
        "-c",
        bootstrap_source,
        "--launcher-fd",
        str(descriptors["launcher"]),
        "--expected-launcher-sha256",
        expected_launcher_sha256,
        "--bootstrap-source-sha256",
        digest(bootstrap_source.encode("utf-8")),
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
    option_names = {
        "contract": "--contract-fd",
        "oracle_code": "--oracle-code-fd",
        "oracle_contract": "--oracle-contract-fd",
        "oracle_schema": "--oracle-schema-fd",
        "reference_code": "--reference-code-fd",
        "reference_contract": "--reference-contract-fd",
    }
    for role, option in option_names.items():
        if role in descriptors:
            arguments.extend((option, str(descriptors[role])))
    try:
        if mutate_after_open is not None:
            mutate_after_open()
        return subprocess.run(
            arguments,
            cwd=ROOT,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "LANG": "C.UTF-8",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            pass_fds=tuple(descriptors.values()),
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        for descriptor in reversed(tuple(descriptors.values())):
            os.close(descriptor)


def build_verifier_request(
    state_root: Path,
    oracle_request: dict,
    runtime: dict[str, object],
) -> tuple[Path, dict[str, Path]]:
    input_dir = state_root / "inputs"
    snapshot_paths: dict[str, Path] = {}
    for label, role, suffix in (
        ("oracle_code_snapshot", "oracle_code_bytes", ".py"),
        ("oracle_contract_snapshot", "oracle_contract_bytes", ".json"),
        ("oracle_schema_snapshot", "oracle_schema_bytes", ".json"),
        ("reference_code_snapshot", "reference_code_bytes", ".py"),
        ("reference_contract_snapshot", "reference_contract_bytes", ".json"),
    ):
        target = input_dir / f"{label}{suffix}"
        target.write_bytes(runtime[role])
        snapshot_paths[label] = target
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
            state_root,
            state_root / "inputs" / "oracle_request.json",
            "oracle_request",
        ),
        **{
            label: artifact(state_root, path, label)
            for label, path in snapshot_paths.items()
        },
        "oracle_intent": artifact(
            state_root,
            launch_output_root(state_root, "ORACLE") / "oracle_output" / "intent.json",
            "oracle_intent",
        ),
        "oracle_commit": artifact(
            state_root,
            launch_output_root(state_root, "ORACLE") / "oracle_output" / "COMMIT.json",
            "oracle_commit",
        ),
        "oracle_ledger": artifact(
            state_root,
            launch_output_root(state_root, "ORACLE")
            / "oracle_output"
            / "oracle_ledger.jsonl",
            "oracle_ledger",
        ),
        "oracle_manifest": artifact(
            state_root,
            launch_output_root(state_root, "ORACLE")
            / "oracle_output"
            / "oracle_manifest.json",
            "oracle_manifest",
        ),
        "output_directory": "verifier_output",
    }
    return write_json(input_dir / "verifier_request.json", request), snapshot_paths


def build_pure_verifier_case(root: Path) -> dict[str, object]:
    state_root = root / "state"
    oracle_request = build_inputs(state_root)
    oracle_request_path = write_json(
        state_root / "inputs" / "oracle_request.json",
        oracle_request,
    )
    runtime = runtime_copies(root)
    oracle_completed = invoke_launcher(
        runtime,
        operation="ORACLE",
        request_path=oracle_request_path,
        state_root=state_root,
    )
    assert oracle_completed.returncode == 0, (
        oracle_completed.stdout + oracle_completed.stderr
    )
    request_path, snapshot_paths = build_verifier_request(
        state_root,
        oracle_request,
        runtime,
    )
    request_bytes = request_path.read_bytes()
    oracle_release = types.MappingProxyType({
        "code_bytes": runtime["oracle_code_bytes"],
        "contract_bytes": runtime["oracle_contract_bytes"],
        "schema_bytes": runtime["oracle_schema_bytes"],
    })
    input_root_fd = os.open(
        state_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        request, artifact_blobs, raw_artifacts = (
            launcher._snapshot_verifier_request_artifacts(
                request_bytes,
                input_root_fd,
                oracle_release_blobs=oracle_release,
                reference_code_bytes=runtime["reference_code_bytes"],
                reference_contract_bytes=runtime["reference_contract_bytes"],
            )
        )
    finally:
        os.close(input_root_fd)
    reference_boundary = launcher._AuditBoundary()
    reference_result_bytes, reference_result_sha256 = (
        launcher._run_reference_snapshot(
            runtime["reference_code_bytes"],
            raw_artifacts,
            reference_boundary,
        )
    )
    launcher_sha256 = digest(runtime["launcher_bytes"])
    verifier_boundary = launcher._AuditBoundary()
    namespace = launcher._compile_and_load(
        "VERIFIER",
        runtime["verifier_code_bytes"],
        runtime["verifier_schema_bytes"],
        launcher_sha256,
        None,
        verifier_boundary,
    )
    target = launcher._pure_verifier_target(namespace)
    reference_attestation = tuple(sorted({
        "reference_code_sha256": launcher.PINNED_RELEASES["REFERENCE"][
            "code_sha256"
        ],
        "reference_contract_sha256": launcher.PINNED_RELEASES["REFERENCE"][
            "contract_sha256"
        ],
        "reference_result_sha256": reference_result_sha256,
    }.items()))
    returned = target(
        request_bytes,
        artifact_blobs,
        tuple(sorted(oracle_release.items())),
        reference_result_bytes,
        reference_attestation,
    )
    return {
        "state_root": state_root,
        "request": request,
        "request_path": request_path,
        "request_bytes": request_bytes,
        "runtime": runtime,
        "snapshot_paths": snapshot_paths,
        "oracle_release": oracle_release,
        "artifact_blobs": artifact_blobs,
        "reference_result_bytes": reference_result_bytes,
        "reference_result_sha256": reference_result_sha256,
        "launcher_sha256": launcher_sha256,
        "returned": returned,
    }


def reseal_verifier_output(
    receipt: dict[str, object],
    commit: dict[str, object],
) -> tuple[bytes, bytes]:
    receipt["verifier_receipt_sha256"] = launcher._embedded_sha256(
        receipt,
        "verifier_receipt_sha256",
    )
    receipt_bytes = launcher._canonical_bytes(receipt) + b"\n"
    commit["receipt_sha256"] = digest(receipt_bytes)
    commit["receipt_size_bytes"] = len(receipt_bytes)
    commit["verifier_receipt_sha256"] = receipt["verifier_receipt_sha256"]
    return receipt_bytes, launcher._canonical_bytes(commit) + b"\n"


def test_native_rename_exclusive_preserves_existing_destination(tmp_path: Path) -> None:
    root = tmp_path / "rename-root"
    root.mkdir(mode=0o700)
    source = root / "source"
    destination = root / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "source.txt").write_text("source", encoding="utf-8")
    (destination / "destination.txt").write_text("destination", encoding="utf-8")
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        rename_exclusive = launcher._build_native_rename_exclusive()
        with pytest.raises(FileExistsError):
            rename_exclusive(root_fd, "source", "destination")
    finally:
        os.close(root_fd)
    assert (source / "source.txt").read_text(encoding="utf-8") == "source"
    assert (destination / "destination.txt").read_text(encoding="utf-8") == "destination"


def test_unsupported_native_rename_has_no_sealed_fallback() -> None:
    with pytest.raises(launcher.LauncherError, match="requires Darwin"):
        launcher._build_native_rename_exclusive(platform_name="linux")
    with pytest.raises(launcher.LauncherError, match="unavailable"):
        launcher._build_native_rename_exclusive(platform_name="darwin", library=object())


def test_direct_path_loaded_launcher_cannot_claim_bootstrap_attestation() -> None:
    descriptor = _open_readonly(ROOT / "paper_research_fd_launcher_v2.py")
    try:
        with pytest.raises(launcher.LauncherError, match="not compiled by"):
            launcher._validate_bootstrap(descriptor)
    finally:
        os.close(descriptor)


def test_launcher_argument_error_is_structured_and_help_never_succeeds(
    capfd: pytest.CaptureFixture[str],
) -> None:
    assert launcher.main(["--help"]) == 2
    captured = capfd.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload == {
        "error_code": "SEALED_FD_LAUNCHER_FAIL_CLOSED",
        "error_sha256": payload["error_sha256"],
        "ok": False,
    }


def test_launcher_write_all_handles_partial_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written = bytearray()

    def partial_write(descriptor: int, value: bytes) -> int:
        assert descriptor == 99
        count = min(3, len(value))
        written.extend(value[:count])
        return count

    monkeypatch.setattr(launcher.os, "write", partial_write)
    launcher._write_all(99, b"0123456789")
    assert bytes(written) == b"0123456789"


def test_reference_snapshot_requires_readonly_regular_fd(tmp_path: Path) -> None:
    source = tmp_path / "reference.py"
    source.write_bytes(b"pinned bytes")
    writable_fd = os.open(source, os.O_RDWR)
    directory_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(launcher.LauncherError, match="must be read-only"):
            launcher._snapshot_readonly_regular_fd(
                writable_fd,
                "reference code",
                maximum_bytes=launcher.MAX_RUNTIME_ARTIFACT_BYTES,
            )
        with pytest.raises(launcher.LauncherError, match="regular file"):
            launcher._snapshot_readonly_regular_fd(
                directory_fd,
                "reference code",
                maximum_bytes=launcher.MAX_RUNTIME_ARTIFACT_BYTES,
            )
    finally:
        os.close(directory_fd)
        os.close(writable_fd)


def test_launcher_acl_guards_reject_roots_artifacts_and_allow_xattrs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "acl-root"
    root.mkdir(mode=0o700)
    artifact_path = root / "artifact.bin"
    artifact_path.write_bytes(b"sealed bytes")
    artifact_path.chmod(0o600)
    add_test_xattr(root)
    add_test_xattr(artifact_path)
    root_fd = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    artifact_fd = os.open(artifact_path, os.O_RDONLY)
    try:
        launcher._validate_root_fd(root_fd, "ACL fixture root")
        assert launcher._snapshot_readonly_regular_fd(
            artifact_fd, "ACL fixture artifact", maximum_bytes=1024
        ) == b"sealed bytes"
        add_test_acl(artifact_path)
        with pytest.raises(launcher.LauncherError, match="extended ACL"):
            launcher._snapshot_readonly_regular_fd(
                artifact_fd, "ACL fixture artifact", maximum_bytes=1024
            )
        clear_test_acl(artifact_path)
        add_test_acl(root)
        with pytest.raises(launcher.LauncherError, match="extended ACL"):
            launcher._validate_root_fd(root_fd, "ACL fixture root")
    finally:
        os.close(artifact_fd)
        os.close(root_fd)


def test_launcher_acl_api_fails_closed_on_unsupported_and_api_errors(
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
            launcher.ctypes.set_errno(self.error_number)
            return self.result

    with pytest.raises(launcher.LauncherError, match="unavailable on this host"):
        launcher._bind_launcher_extended_acl_api("linux", object())
    with pytest.raises(launcher.LauncherError, match="API is unavailable"):
        launcher._bind_launcher_extended_acl_api("darwin", object())

    path = tmp_path / "acl-api-file"
    path.write_bytes(b"x")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        get = FakeCall(None, launcher.errno.EOPNOTSUPP)
        free = FakeCall(0)
        monkeypatch.setattr(launcher, "_LAUNCHER_ACL_GET_FD_NP", get)
        monkeypatch.setattr(launcher, "_LAUNCHER_ACL_FREE", free)
        with pytest.raises(launcher.LauncherError, match="inspection failed"):
            launcher._require_launcher_no_extended_acl_fd(
                descriptor, "fake ACL"
            )

        get = FakeCall(1)
        free = FakeCall(-1, launcher.errno.EIO)
        monkeypatch.setattr(launcher, "_LAUNCHER_ACL_GET_FD_NP", get)
        monkeypatch.setattr(launcher, "_LAUNCHER_ACL_FREE", free)
        with pytest.raises(launcher.LauncherError, match="release failed"):
            launcher._require_launcher_no_extended_acl_fd(
                descriptor, "fake ACL"
            )
        assert free.calls == 1
    finally:
        os.close(descriptor)


def test_launcher_acl_guards_cover_relative_and_published_children(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    parent = input_root / "nested"
    parent.mkdir(parents=True, mode=0o700)
    artifact = parent / "artifact.json"
    artifact.write_bytes(b"{}\n")
    artifact.chmod(0o600)
    input_fd = os.open(
        input_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        add_test_acl(parent)
        with pytest.raises(launcher.LauncherError, match="extended ACL"):
            launcher._snapshot_relative_artifact(
                input_fd,
                "nested/artifact.json",
                "relative ACL fixture",
                expected_size=3,
                maximum_bytes=1024,
            )
    finally:
        os.close(input_fd)

    output_root = tmp_path / "output"
    output_root.mkdir(mode=0o700)
    output_name = "result"
    lock = output_root / f".{output_name}.lock"
    lock.write_bytes(b"lock\n")
    lock.chmod(0o600)
    output = output_root / output_name
    output.mkdir(mode=0o700)
    result_file = output / "COMMIT.json"
    result_file.write_bytes(b"{}\n")
    result_file.chmod(0o600)
    output_fd = os.open(
        output_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        launcher._validate_runtime_output_acl_tree(
            output_fd,
            output_name,
            frozenset({"COMMIT.json"}),
            "published ACL fixture",
        )
        for target in (lock, output, result_file):
            add_test_acl(target)
            try:
                with pytest.raises(launcher.LauncherError, match="extended ACL"):
                    launcher._validate_runtime_output_acl_tree(
                        output_fd,
                        output_name,
                        frozenset({"COMMIT.json"}),
                        "published ACL fixture",
                    )
            finally:
                clear_test_acl(target)
    finally:
        os.close(output_fd)


def test_launcher_rejects_acl_inherited_by_new_stage_file(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    add_test_acl(stage, inherited=True)
    stage_fd = os.open(
        stage,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        with pytest.raises(launcher.LauncherError, match="extended ACL"):
            launcher._write_immutable_file_at(stage_fd, "receipt.json", b"{}\n")
    finally:
        os.close(stage_fd)


def test_audit_boundary_allows_one_pinned_exec_and_denies_escape_events() -> None:
    boundary = launcher._AuditBoundary()
    boundary.begin_runtime_phase()
    source = b"value = 1"
    boundary.allow_exact_compile(source, "<sealed-test>")
    boundary.hook("compile", (source, "<sealed-test>"))
    pinned = compile(source, "<sealed-test>", "exec")
    boundary.allow_exact_exec(pinned)
    boundary.hook("exec", (pinned,))
    with pytest.raises(launcher.LauncherError, match="dynamic compile"):
        boundary.hook("compile", (b"value = 2", "<sealed-test>"))
    with pytest.raises(launcher.LauncherError, match="dynamic exec"):
        boundary.hook("exec", (compile("value = 2", "<string>", "exec"),))
    for event in (
        "socket.connect",
        "subprocess.Popen",
        "ctypes.dlopen",
        "code.__new__",
        "function.__new__",
        "marshal.loads",
        "os.system",
        "os.posix_spawn",
        "os.exec",
    ):
        with pytest.raises(launcher.LauncherError, match="capability denied"):
            boundary.hook(event, ())
    boundary.end_runtime_phase()


def test_reference_release_pins_are_literal_and_exact() -> None:
    assert launcher.PINNED_RELEASES["REFERENCE"] == {
        "code_sha256": (
            "cbac8e308bc11cd334f1cd23d23e4e75019074c1bdcfb66873b9254e3d6d520f"
        ),
        "contract_sha256": (
            "276c34f4174a15d188406ef870d86a8d0bcbbc1b64b1f45381a033e20eb5d8f5"
        ),
    }
    assert digest(
        (ROOT / "paper_research_double_entry_reference_v2.py").read_bytes()
    ) == launcher.PINNED_RELEASES["REFERENCE"]["code_sha256"]
    assert digest(
        (ROOT / "PAPER_RESEARCH_DOUBLE_ENTRY_REFERENCE_CONTRACT_V2.json").read_bytes()
    ) == launcher.PINNED_RELEASES["REFERENCE"]["contract_sha256"]


def test_reference_loader_uses_fresh_module_and_seals_imports() -> None:
    boundary = launcher._AuditBoundary()
    source = (ROOT / "paper_research_double_entry_reference_v2.py").read_bytes()
    launcher._scan_reference_source(source, "<sealed-reference-v2>")
    boundary.begin_reference_phase()
    try:
        module, replay = launcher._compile_and_load_reference(source, boundary)
        assert module.__name__ == "_sealed_double_entry_reference_v2"
        assert module.__name__ not in sys.modules
        assert replay is module.__dict__["replay_reference"]
        assert replay.__globals__ is module.__dict__
        assert launcher._pure_reference_target(module.__dict__) is replay
        assert set(module.__dict__["__builtins__"]) == launcher.PURE_REFERENCE_BUILTINS
        assert not (
            set(module.__dict__["__builtins__"])
            & launcher.REFERENCE_INITIALIZATION_BUILTINS
        )
        assert "__import__" not in module.__dict__["__builtins__"]
    finally:
        module.__dict__.clear()
        boundary.end_reference_phase()


def test_reference_source_open_probe_cannot_read_non_input_file(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "not-an-input.txt"
    probe.write_text("REFERENCE_ESCAPE_SECRET", encoding="utf-8")
    source = (
        (ROOT / "paper_research_double_entry_reference_v2.py").read_bytes()
        + b"\nraise RuntimeError(open("
        + repr(str(probe)).encode("utf-8")
        + b").read())\n"
    )
    boundary = launcher._AuditBoundary()
    with pytest.raises(launcher.LauncherError, match="forbidden capability") as caught:
        launcher._run_reference_snapshot(
            source,
            types.MappingProxyType({}),
            boundary,
        )
    assert "REFERENCE_ESCAPE_SECRET" not in str(caught.value)
    assert boundary._reference_phase_active is False


@pytest.mark.parametrize(
    "suffix",
    (
        b"\nescape = ENGINE_ID.__class__\n",
        b"\nescape = __import__('os')\n",
        b"\nimport os\n",
        b"\ng = getattr\n",
        b"\nescape = getattr(ENGINE_ID, '__class__')\n",
        b"\ndef nested_escape():\n    import pathlib\n",
    ),
)
def test_reference_source_scan_rejects_reflection_import_and_getattr_aliases(
    suffix: bytes,
) -> None:
    source = (
        ROOT / "paper_research_double_entry_reference_v2.py"
    ).read_bytes() + suffix
    with pytest.raises(launcher.LauncherError):
        launcher._run_reference_snapshot(
            source,
            types.MappingProxyType({}),
            launcher._AuditBoundary(),
        )


def test_reference_source_scan_rejects_cost_field_walrus_rebinding() -> None:
    source = (ROOT / "paper_research_double_entry_reference_v2.py").read_bytes()
    original = b"any(getattr(raw_arm, field) != 0 for field in cost_fields)"
    mutant = (
        b"any(((field := '_' * 2 + 'class' + '_' * 2) and "
        b"getattr(raw_arm, field)) for field in cost_fields)"
    )
    assert source.count(original) == 1
    with pytest.raises(launcher.LauncherError, match="predicate|rebinding"):
        launcher._run_reference_snapshot(
            source.replace(original, mutant),
            types.MappingProxyType({}),
            launcher._AuditBoundary(),
        )


def test_reference_audit_scope_blocks_open_then_becomes_dormant(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "audit-probe.txt"
    probe.write_text("outside-scope-readable", encoding="utf-8")
    boundary = launcher._AuditBoundary()
    sys.addaudithook(boundary.hook)
    boundary.begin_reference_phase()
    try:
        with pytest.raises(launcher.LauncherError, match="reference replay capability"):
            open(probe, "rb")
        for event in (
            "os.listdir",
            "os.getenv",
            "time.time",
            "random.seed",
            "uuid.uuid4",
            "subprocess.Popen",
            "socket.connect",
            "ctypes.dlopen",
        ):
            with pytest.raises(launcher.LauncherError, match="capability denied"):
                boundary.hook(event, ())
    finally:
        boundary.end_reference_phase()
    assert probe.read_text(encoding="utf-8") == "outside-scope-readable"


def test_dataclass_codegen_exception_is_denied_outside_exact_stdlib_stack() -> None:
    boundary = launcher._AuditBoundary()
    boundary.begin_reference_phase()
    boundary.allow_dataclass_codegen()
    with pytest.raises(launcher.LauncherError, match="dynamic compile"):
        boundary.hook("compile", (b"return None", "<string>"))
    with pytest.raises(launcher.LauncherError, match="dynamic exec"):
        boundary.hook("exec", (compile("pass", "<string>", "exec"),))
    boundary.seal_dataclass_codegen()
    boundary.end_reference_phase()


def test_reference_namespace_is_destroyed_even_when_result_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("_test_reference")
    module.sentinel = object()

    def invalid_replay(_: object) -> dict[str, object]:
        return {}

    monkeypatch.setattr(
        launcher,
        "_compile_and_load_reference",
        lambda *_: (module, invalid_replay),
    )
    monkeypatch.setattr(launcher, "_scan_reference_source", lambda *_: None)
    monkeypatch.setattr(launcher, "_pure_reference_target", lambda *_: invalid_replay)
    with pytest.raises(launcher.LauncherError, match="reference result schema"):
        launcher._run_reference_snapshot(
            b"unused",
            types.MappingProxyType({}),
            launcher._AuditBoundary(),
        )
    assert module.__dict__ == {}


@pytest.mark.parametrize(
    "forbidden_global",
    (
        "_SEALED_REFERENCE_REPLAY",
        "_SEALED_REFERENCE_RESULT_BYTES",
        "_SEALED_REFERENCE_MODULE",
        "_SEALED_RENAME_EXCLUSIVE",
    ),
)
def test_verifier_initialization_rejects_live_reference_or_publication_global(
    forbidden_global: str,
) -> None:
    boundary = launcher._AuditBoundary()
    with pytest.raises(launcher.LauncherError, match="forbidden live capability"):
        launcher._compile_and_load(
            "VERIFIER",
            f"{forbidden_global} = None\n".encode("ascii"),
            b"{}",
            "0" * 64,
            None,
            boundary,
        )


def test_real_verifier_namespace_has_only_immutable_launcher_injections() -> None:
    boundary = launcher._AuditBoundary()
    namespace = launcher._compile_and_load(
        "VERIFIER",
        (ROOT / "paper_research_oracle_verifier_v2.py").read_bytes(),
        (ROOT / "paper_research_oracle_verifier_schema_v2.json").read_bytes(),
        "0" * 64,
        None,
        boundary,
    )
    target = launcher._pure_verifier_target(namespace)
    assert target is namespace["verify_sealed_bytes"]
    assert target.__closure__ is None
    assert target.__code__.co_varnames[:5] == (
        "request_bytes",
        "artifact_blobs",
        "oracle_release_blobs",
        "reference_result_bytes",
        "reference_attestation",
    )
    assert {
        name for name in namespace if name.startswith("_SEALED_")
    } == {
        "_SEALED_RUNTIME_CODE_BYTES",
        "_SEALED_RUNTIME_CODE_SHA256",
        "_SEALED_SCHEMA_BYTES",
        "_SEALED_SCHEMA_SHA256",
        "_SEALED_LAUNCHER_SHA256",
        "_SEALED_RUNTIME",
    }
    assert set(namespace["__builtins__"]) == launcher.PURE_VERIFIER_BUILTINS
    assert not (
        set(namespace["__builtins__"])
        & launcher.VERIFIER_INITIALIZATION_BUILTINS
    )
    assert not any(
        type(value) is types.MethodType for value in namespace.values()
    )
    assert type(namespace["SUPPORTED_REFERENCE_RELEASE"]) is types.MappingProxyType
    assert type(namespace["FORBIDDEN_PROPOSAL_TOKENS"]) is frozenset
    with pytest.raises(TypeError):
        namespace["SUPPORTED_REFERENCE_RELEASE"]["code_sha256"] = "0" * 64


@pytest.mark.parametrize(
    "source",
    (
        b"_ = value.__class__\nreturn (b'{}\\n', b'{}\\n')",
        b"_ = '__subclasses__'\nreturn (b'{}\\n', b'{}\\n')",
        b"_ = globals()\nreturn (b'{}\\n', b'{}\\n')",
        b"_ = getattr(value, 'field', None)\nreturn (b'{}\\n', b'{}\\n')",
        b"_ = value.__traceback__\nreturn (b'{}\\n', b'{}\\n')",
        b"_ = value.f_back\nreturn (b'{}\\n', b'{}\\n')",
    ),
)
def test_pure_verifier_reachable_scan_rejects_reflection(source: bytes) -> None:
    boundary = launcher._AuditBoundary()
    namespace = launcher._compile_and_load(
        "VERIFIER",
        pure_verifier_source(source),
        b"{}",
        "0" * 64,
        None,
        boundary,
    )
    with pytest.raises(launcher.LauncherError, match="forbidden capability"):
        launcher._pure_verifier_target(namespace)


def test_pure_verifier_reachable_scan_rejects_post_seal_import() -> None:
    boundary = launcher._AuditBoundary()
    namespace = launcher._compile_and_load(
        "VERIFIER",
        pure_verifier_source(
            b"import json\nreturn (b'{}\\n', b'{}\\n')"
        ),
        b"{}",
        "0" * 64,
        None,
        boundary,
    )
    with pytest.raises(launcher.LauncherError, match="dynamic code capability"):
        launcher._pure_verifier_target(namespace)


def test_verifier_cannot_retain_initialization_importer() -> None:
    boundary = launcher._AuditBoundary()
    namespace = launcher._compile_and_load(
        "VERIFIER",
        (
            b"captured_import = __builtins__['__import__']\n"
            + pure_verifier_source()
        ),
        b"{}",
        "0" * 64,
        None,
        boundary,
    )
    assert type(namespace["captured_import"]) is types.MethodType
    with pytest.raises(launcher.LauncherError, match="closure capability"):
        launcher._pure_verifier_target(namespace)


def test_pure_verifier_signature_cannot_accept_output_callback() -> None:
    boundary = launcher._AuditBoundary()
    source = (
        b"def verify_sealed_bytes(request_bytes, artifact_blobs, "
        b"oracle_release_blobs, reference_result_bytes, "
        b"reference_attestation, publish):\n"
        b"    return (b'{}\\n', b'{}\\n')\n"
    )
    namespace = launcher._compile_and_load(
        "VERIFIER", source, b"{}", "0" * 64, None, boundary
    )
    with pytest.raises(launcher.LauncherError, match="signature is not exact"):
        launcher._pure_verifier_target(namespace)


@pytest.mark.parametrize(
    ("source", "case"),
    (
        (b"import importlib\n", "importlib statement"),
        (
            b"__builtins__['__import__']('importlib')\n",
            "injected builtins import",
        ),
        (b"import _posixsubprocess\n", "private subprocess module"),
        (
            b"__builtins__['__import__']('_posixsubprocess')\n",
            "injected builtins private subprocess import",
        ),
    ),
)
def test_runtime_import_escape_is_rejected_before_entrypoint_resolution(
    source: bytes,
    case: str,
) -> None:
    boundary = launcher._AuditBoundary()
    with pytest.raises(
        launcher.LauncherError,
        match="outside the frozen dependency set",
    ):
        launcher._compile_and_load(
            "VERIFIER",
            source,
            b"{}",
            "0" * 64,
            None,
            boundary,
        )


def test_runtime_import_is_absent_after_initialization() -> None:
    boundary = launcher._AuditBoundary()
    namespace = launcher._compile_and_load(
        "VERIFIER",
        b"import json\n" + pure_verifier_source(),
        b"{}",
        "0" * 64,
        None,
        boundary,
    )
    assert namespace["json"] is json
    assert "__import__" not in namespace["__builtins__"]
    for module_name in ("json", "importlib", "_posixsubprocess"):
        with pytest.raises(launcher.LauncherError, match="dynamic import capability denied"):
            boundary.sealed_import(module_name)


def test_launcher_strictly_rejects_pure_verifier_return_tampering(
    tmp_path: Path,
) -> None:
    case = build_pure_verifier_case(tmp_path)
    receipt_bytes, commit_bytes = case["returned"]
    validation = {
        "request_bytes": case["request_bytes"],
        "artifact_blobs": case["artifact_blobs"],
        "reference_result_bytes": case["reference_result_bytes"],
        "reference_result_sha256": case["reference_result_sha256"],
        "launcher_sha256": case["launcher_sha256"],
    }
    assert launcher._validate_verifier_return(
        (receipt_bytes, commit_bytes),
        **validation,
    )[:2] == (receipt_bytes, commit_bytes)
    with pytest.raises(launcher.LauncherError, match="exact byte pair"):
        launcher._validate_verifier_return(
            [receipt_bytes, commit_bytes],
            **validation,
        )
    with pytest.raises(launcher.LauncherError, match="not canonical JSON"):
        launcher._validate_verifier_return(
            (receipt_bytes[:-1] + b" \n", commit_bytes),
            **validation,
        )

    original_receipt = json.loads(receipt_bytes)
    original_commit = json.loads(commit_bytes)
    semantic_mutations = (
        (
            "receipt schema",
            lambda receipt, _: receipt.__setitem__("unexpected", 1),
        ),
        (
            "identity or classification",
            lambda receipt, _: receipt.__setitem__("classification", "ADMISSIBLE"),
        ),
        (
            "boolean authority gate",
            lambda receipt, _: receipt.__setitem__("admission_eligible", True),
        ),
        (
            "authority or terminal state",
            lambda receipt, _: receipt["authority"].__setitem__(
                "live_authority", True
            ),
        ),
        (
            "input artifact binding",
            lambda receipt, _: receipt["input_artifact_sha256"].__setitem__(
                "proposal", "0" * 64
            ),
        ),
        (
            "reference result binding",
            lambda receipt, _: receipt.__setitem__(
                "reference_journal_transaction_count",
                receipt["reference_journal_transaction_count"] + 1,
            ),
        ),
        (
            "release or self binding",
            lambda receipt, _: receipt["verifier_release_content_binding"].__setitem__(
                "reference_result_sha256", "0" * 64
            ),
        ),
        (
            "COMMIT binding",
            lambda _, commit: commit.__setitem__("request_sha256", "0" * 64),
        ),
    )
    for expected_error, mutate in semantic_mutations:
        receipt = json.loads(json.dumps(original_receipt))
        commit = json.loads(json.dumps(original_commit))
        mutate(receipt, commit)
        mutated_receipt, mutated_commit = reseal_verifier_output(receipt, commit)
        with pytest.raises(launcher.LauncherError, match=expected_error):
            launcher._validate_verifier_return(
                (mutated_receipt, mutated_commit),
                **validation,
            )


def test_verifier_descriptor_matrix_and_snapshot_path_toctou(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = build_pure_verifier_case(tmp_path)
    state_root = case["state_root"]
    request = case["request"]
    runtime = case["runtime"]
    oracle_release = case["oracle_release"]

    def snapshot(candidate: dict[str, object]) -> tuple[object, ...]:
        request_bytes = launcher._canonical_bytes(candidate) + b"\n"
        root_fd = os.open(
            state_root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            return launcher._snapshot_verifier_request_artifacts(
                request_bytes,
                root_fd,
                oracle_release_blobs=oracle_release,
                reference_code_bytes=runtime["reference_code_bytes"],
                reference_contract_bytes=runtime["reference_contract_bytes"],
            )
        finally:
            os.close(root_fd)

    missing = json.loads(json.dumps(request))
    missing.pop("accounting_policy")
    with pytest.raises(launcher.LauncherError, match="request schema"):
        snapshot(missing)

    swapped = json.loads(json.dumps(request))
    swapped["accounting_policy"], swapped["evaluation_policy"] = (
        swapped["evaluation_policy"],
        swapped["accounting_policy"],
    )
    with pytest.raises(launcher.LauncherError, match="descriptor identity"):
        snapshot(swapped)

    aliased = json.loads(json.dumps(request))
    aliased["evaluation_policy"] = {
        **aliased["accounting_policy"],
        "artifact_id": "evaluation_policy",
    }
    with pytest.raises(launcher.LauncherError, match="alias each other"):
        snapshot(aliased)

    proposal_path = state_root / request["proposal"]["relative_path"]
    original_proposal = proposal_path.read_bytes()
    proposal_path.write_bytes(b"x" * len(original_proposal))
    with pytest.raises(launcher.LauncherError, match="exact-byte binding"):
        snapshot(request)
    proposal_path.write_bytes(original_proposal)

    original_inode = proposal_path.stat().st_ino
    replacement = proposal_path.with_suffix(".replacement")
    real_pread = launcher.os.pread
    replaced = False

    def replace_after_open(descriptor: int, count: int, offset: int) -> bytes:
        nonlocal replaced
        if not replaced and os.fstat(descriptor).st_ino == original_inode:
            replacement.write_bytes(b"path replacement is not snapshotted")
            os.replace(replacement, proposal_path)
            replaced = True
        return real_pread(descriptor, count, offset)

    monkeypatch.setattr(launcher.os, "pread", replace_after_open)
    with pytest.raises(launcher.LauncherError, match="changed during snapshot"):
        snapshot(request)
    assert replaced is True


def test_launcher_publication_is_idempotent_and_detects_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = build_pure_verifier_case(tmp_path / "case")
    receipt_bytes, commit_bytes = case["returned"]
    output_root = tmp_path / "publication"
    output_root.mkdir(mode=0o700)
    output_root_fd = os.open(
        output_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    rename_exclusive = launcher._build_native_rename_exclusive()
    try:
        launcher._publish_verifier_bytes(
            output_root_fd,
            "verified",
            receipt_bytes,
            commit_bytes,
            rename_exclusive,
        )
        launcher._publish_verifier_bytes(
            output_root_fd,
            "verified",
            receipt_bytes,
            commit_bytes,
            rename_exclusive,
        )
        assert (output_root / "verified" / "verifier_receipt.json").read_bytes() \
            == receipt_bytes
        assert (output_root / "verified" / "COMMIT.json").read_bytes() \
            == commit_bytes

        original_validate = launcher._validate_verifier_output_directory
        replaced = False

        def replace_after_validation(
            directory_fd: int,
            expected_receipt: bytes,
            expected_commit: bytes,
        ) -> None:
            nonlocal replaced
            original_validate(directory_fd, expected_receipt, expected_commit)
            if not replaced:
                os.rename(
                    "verified",
                    "verified-held",
                    src_dir_fd=output_root_fd,
                    dst_dir_fd=output_root_fd,
                )
                os.mkdir("verified", 0o700, dir_fd=output_root_fd)
                replaced = True

        monkeypatch.setattr(
            launcher,
            "_validate_verifier_output_directory",
            replace_after_validation,
        )
        with pytest.raises(launcher.LauncherError, match="pathname changed"):
            launcher._publish_verifier_bytes(
                output_root_fd,
                "verified",
                receipt_bytes,
                commit_bytes,
                rename_exclusive,
            )
        assert replaced is True
    finally:
        os.close(output_root_fd)


def test_launcher_rejects_same_inode_input_and_output_roots(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    oracle_request = build_inputs(state_root)
    request_path = write_json(
        state_root / "inputs" / "oracle_request.json",
        oracle_request,
    )
    runtime = runtime_copies(tmp_path)
    completed = invoke_launcher(
        runtime,
        operation="ORACLE",
        request_path=request_path,
        state_root=state_root,
        output_root=state_root,
    )
    assert_launcher_failure(
        completed,
        "input and output roots must be distinct directory inodes",
    )
    assert not (state_root / "oracle_output").exists()


@pytest.mark.parametrize(
    "python_flags",
    (
        ("-S", "-B"),
        ("-I", "-B"),
        ("-I", "-S"),
    ),
)
def test_bootstrap_rejects_missing_I_S_or_B_before_output(
    tmp_path: Path,
    python_flags: tuple[str, ...],
) -> None:
    state_root = tmp_path / "state"
    oracle_request = build_inputs(state_root)
    request_path = write_json(state_root / "inputs" / "oracle_request.json", oracle_request)
    runtime = runtime_copies(tmp_path)
    completed = invoke_launcher(
        runtime,
        operation="ORACLE",
        request_path=request_path,
        state_root=state_root,
        python_flags=python_flags,
    )
    assert completed.returncode == 120
    assert json.loads(completed.stdout)["error_code"] == (
        "SEALED_FD_BOOTSTRAP_FAIL_CLOSED"
    )
    assert not (
        launch_output_root(state_root, "ORACLE") / "oracle_output"
    ).exists()


def test_bootstrap_rejects_unexpected_sys_path_before_launcher_exec(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    oracle_request = build_inputs(state_root)
    request_path = write_json(state_root / "inputs" / "oracle_request.json", oracle_request)
    runtime = runtime_copies(tmp_path)
    modified_bootstrap = (
        "import sys\nsys.path.append('/tmp/unexpected-site-packages')\n"
        + launcher.FIXED_BOOTSTRAP_SOURCE
    )
    completed = invoke_launcher(
        runtime,
        operation="ORACLE",
        request_path=request_path,
        state_root=state_root,
        bootstrap_source=modified_bootstrap,
    )
    assert completed.returncode == 120
    assert json.loads(completed.stdout)["error_code"] == (
        "SEALED_FD_BOOTSTRAP_FAIL_CLOSED"
    )
    assert not (
        launch_output_root(state_root, "ORACLE") / "oracle_output"
    ).exists()


def test_bootstrap_rejects_acl_bearing_inherited_launcher_before_exec(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    oracle_request = build_inputs(state_root)
    request_path = write_json(
        state_root / "inputs" / "oracle_request.json", oracle_request
    )
    runtime = runtime_copies(tmp_path)
    add_test_acl(Path(runtime["launcher"]))
    completed = invoke_launcher(
        runtime,
        operation="ORACLE",
        request_path=request_path,
        state_root=state_root,
    )
    assert completed.returncode == 120
    assert json.loads(completed.stdout)["error_code"] == (
        "SEALED_FD_BOOTSTRAP_FAIL_CLOSED"
    )
    assert not (
        launch_output_root(state_root, "ORACLE") / "oracle_output"
    ).exists()


def test_launcher_rejects_unreviewed_bootstrap_source_even_if_caller_hashes_it(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    oracle_request = build_inputs(state_root)
    request_path = write_json(state_root / "inputs" / "oracle_request.json", oracle_request)
    runtime = runtime_copies(tmp_path)
    modified_bootstrap = launcher.FIXED_BOOTSTRAP_SOURCE + "\n# unreviewed variant\n"
    completed = invoke_launcher(
        runtime,
        operation="ORACLE",
        request_path=request_path,
        state_root=state_root,
        bootstrap_source=modified_bootstrap,
    )
    assert_launcher_failure(
        completed,
        "outer bootstrap source is not the reviewed fixed source",
    )
    assert not (
        launch_output_root(state_root, "ORACLE") / "oracle_output"
    ).exists()


def test_same_inode_malicious_launcher_cannot_restore_after_preexec_hash_gate(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    oracle_request = build_inputs(state_root)
    request_path = write_json(state_root / "inputs" / "oracle_request.json", oracle_request)
    runtime = runtime_copies(tmp_path)
    reviewed_copy = tmp_path / "reviewed-launcher.py"
    reviewed_copy.write_bytes(runtime["launcher_bytes"])
    marker = tmp_path / "malicious-launcher-executed"
    malicious = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_bytes(b'executed')\n"
        f"Path({str(runtime['launcher'])!r}).write_bytes(Path({str(reviewed_copy)!r}).read_bytes())\n"
    ).encode("utf-8")

    def overwrite_same_inode_after_fd_open() -> None:
        descriptor = os.open(runtime["launcher"], os.O_WRONLY | os.O_TRUNC)
        try:
            view = memoryview(malicious)
            while view:
                written = os.write(descriptor, view)
                assert written > 0
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    completed = invoke_launcher(
        runtime,
        operation="ORACLE",
        request_path=request_path,
        state_root=state_root,
        mutate_after_open=overwrite_same_inode_after_fd_open,
    )
    assert completed.returncode == 120
    assert json.loads(completed.stdout)["error_code"] == (
        "SEALED_FD_BOOTSTRAP_FAIL_CLOSED"
    )
    assert not marker.exists()
    assert Path(runtime["launcher"]).read_bytes() == malicious
    assert not (
        launch_output_root(state_root, "ORACLE") / "oracle_output"
    ).exists()


@pytest.mark.parametrize("mismatched_role", ("code", "contract", "schema"))
def test_oracle_code_contract_schema_mismatch_fails_before_execution(
    tmp_path: Path,
    mismatched_role: str,
) -> None:
    state_root = tmp_path / "state"
    oracle_request = build_inputs(state_root)
    request_path = write_json(state_root / "inputs" / "oracle_request.json", oracle_request)
    runtime = runtime_copies(tmp_path)
    mismatch = tmp_path / f"wrong-{mismatched_role}"
    mismatch.write_bytes(b"{}\n" if mismatched_role != "code" else b"raise SystemExit(0)\n")
    completed = invoke_launcher(
        runtime,
        operation="ORACLE",
        request_path=request_path,
        state_root=state_root,
        overrides={mismatched_role: mismatch},
    )
    assert completed.returncode == 2
    assert json.loads(completed.stdout) == {
        "error_code": "SEALED_FD_LAUNCHER_FAIL_CLOSED",
        "error_sha256": json.loads(completed.stdout)["error_sha256"],
        "ok": False,
    }
    assert not (
        launch_output_root(state_root, "ORACLE") / "oracle_output"
    ).exists()


@pytest.mark.parametrize("mismatched_role", ("code", "schema"))
def test_verifier_code_and_schema_mismatch_fail_before_execution(
    tmp_path: Path,
    mismatched_role: str,
) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir(parents=True, mode=0o700)
    request_path = write_json(state_root / "request.json", {})
    runtime = runtime_copies(tmp_path)
    mismatch = tmp_path / f"wrong-verifier-{mismatched_role}"
    mismatch.write_bytes(b"{}\n" if mismatched_role == "schema" else b"pass\n")
    completed = invoke_launcher(
        runtime,
        operation="VERIFIER",
        request_path=request_path,
        state_root=state_root,
        overrides={mismatched_role: mismatch},
    )
    assert_launcher_failure(
        completed,
        f"verifier {mismatched_role} FD is not the pinned release",
    )
    assert not (
        launch_output_root(state_root, "VERIFIER") / "verifier_output"
    ).exists()


@pytest.mark.parametrize(
    "omit_roles",
    (
        frozenset({"reference_code"}),
        frozenset({"reference_contract"}),
        frozenset({"reference_code", "reference_contract"}),
    ),
)
def test_verifier_requires_both_reference_fds(
    tmp_path: Path,
    omit_roles: frozenset[str],
) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir(parents=True, mode=0o700)
    request_path = write_json(state_root / "request.json", {})
    runtime = runtime_copies(tmp_path)
    completed = invoke_launcher(
        runtime,
        operation="VERIFIER",
        request_path=request_path,
        state_root=state_root,
        omit_roles=omit_roles,
    )
    assert_launcher_failure(
        completed,
        "Verifier operation requires trusted Oracle and reference release FDs",
    )
    assert not (
        launch_output_root(state_root, "VERIFIER") / "verifier_output"
    ).exists()


@pytest.mark.parametrize(
    "omit_roles",
    (
        frozenset(),
        frozenset({"reference_code"}),
        frozenset({"reference_contract"}),
    ),
)
def test_oracle_rejects_reference_fds(
    tmp_path: Path,
    omit_roles: frozenset[str],
) -> None:
    state_root = tmp_path / "state"
    oracle_request = build_inputs(state_root)
    request_path = write_json(
        state_root / "inputs" / "oracle_request.json", oracle_request
    )
    runtime = runtime_copies(tmp_path)
    completed = invoke_launcher(
        runtime,
        operation="ORACLE",
        request_path=request_path,
        state_root=state_root,
        oracle_reference_fds=True,
        omit_roles=omit_roles,
    )
    assert_launcher_failure(
        completed,
        "Oracle operation rejects verifier-only release FDs",
    )
    assert not (
        launch_output_root(state_root, "ORACLE") / "oracle_output"
    ).exists()


@pytest.mark.parametrize("reference_role", ("reference_code", "reference_contract"))
def test_reference_release_rejects_swapped_code_and_contract_fds(
    tmp_path: Path,
    reference_role: str,
) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir(parents=True, mode=0o700)
    request_path = write_json(state_root / "request.json", {})
    runtime = runtime_copies(tmp_path)
    other_role = (
        "reference_contract" if reference_role == "reference_code" else "reference_code"
    )
    completed = invoke_launcher(
        runtime,
        operation="VERIFIER",
        request_path=request_path,
        state_root=state_root,
        overrides={reference_role: runtime[other_role]},
    )
    assert_launcher_failure(
        completed,
        f"reference {reference_role[10:]} FD is not the pinned release",
    )
    with pytest.raises(launcher.LauncherError, match=f"reference {reference_role[10:]}"):
        launcher._require_pinned_bytes(
            "REFERENCE", reference_role[10:], runtime[f"{other_role}_bytes"]
        )


@pytest.mark.parametrize("reference_role", ("reference_code", "reference_contract"))
def test_reference_release_rejects_arbitrary_wrong_bytes(
    tmp_path: Path,
    reference_role: str,
) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir(parents=True, mode=0o700)
    request_path = write_json(state_root / "request.json", {})
    runtime = runtime_copies(tmp_path)
    wrong = tmp_path / f"wrong-{reference_role}"
    wrong.write_bytes(b"pass\n" if reference_role == "reference_code" else b"{}\n")
    completed = invoke_launcher(
        runtime,
        operation="VERIFIER",
        request_path=request_path,
        state_root=state_root,
        overrides={reference_role: wrong},
    )
    assert_launcher_failure(
        completed,
        f"reference {reference_role[10:]} FD is not the pinned release",
    )
    with pytest.raises(launcher.LauncherError, match=f"reference {reference_role[10:]}"):
        launcher._require_pinned_bytes(
            "REFERENCE", reference_role[10:], wrong.read_bytes()
        )


def test_verifier_and_oracle_code_fds_cannot_be_swapped(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir(parents=True, mode=0o700)
    request_path = write_json(state_root / "request.json", {})
    runtime = runtime_copies(tmp_path)
    completed = invoke_launcher(
        runtime,
        operation="VERIFIER",
        request_path=request_path,
        state_root=state_root,
        overrides={
            "code": runtime["oracle_code"],
            "schema": runtime["oracle_schema"],
            "oracle_code": runtime["verifier_code"],
            "oracle_schema": runtime["verifier_schema"],
        },
    )
    assert_launcher_failure(
        completed,
        "verifier code FD is not the pinned release",
    )
    assert not (
        launch_output_root(state_root, "VERIFIER") / "verifier_output"
    ).exists()


@pytest.mark.parametrize("reference_role", ("reference_code", "reference_contract"))
def test_same_inode_reference_mutation_after_fd_open_fails_closed(
    tmp_path: Path,
    reference_role: str,
) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir(parents=True, mode=0o700)
    request_path = write_json(state_root / "request.json", {})
    runtime = runtime_copies(tmp_path)

    def mutate_same_inode() -> None:
        Path(runtime[reference_role]).write_bytes(b"same-inode reference mutation\n")

    completed = invoke_launcher(
        runtime,
        operation="VERIFIER",
        request_path=request_path,
        state_root=state_root,
        mutate_after_open=mutate_same_inode,
    )
    assert_launcher_failure(
        completed,
        f"reference {reference_role[10:]} FD is not the pinned release",
    )
    assert not (
        launch_output_root(state_root, "VERIFIER") / "verifier_output"
    ).exists()


def test_actual_restricted_bootstrap_oracle_and_pure_verifier_e2e_survives_path_replacement(
    tmp_path: Path,
) -> None:
    cache_snapshot_before = {
        path.relative_to(ROOT).as_posix(): (
            digest(path.read_bytes()),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for directory in ROOT.rglob("__pycache__")
        for path in directory.iterdir()
        if path.is_file()
    }
    state_root = tmp_path / "state"
    oracle_request = build_inputs(state_root)
    oracle_request_path = write_json(
        state_root / "inputs" / "oracle_request.json", oracle_request
    )
    runtime = runtime_copies(tmp_path)
    launcher_source = runtime["launcher_bytes"].decode("utf-8")
    assert "verify_from_fds" not in launcher_source
    assert "_SEALED_REFERENCE_REPLAY" not in launcher_source

    def replace_oracle_runtime_path() -> None:
        launcher_replacement = Path(runtime["launcher"]).with_suffix(".replacement")
        launcher_replacement.write_bytes(
            b"raise RuntimeError('launcher path replacement must not execute')\n"
        )
        os.replace(launcher_replacement, runtime["launcher"])
        replacement = Path(runtime["oracle_code"]).with_suffix(".replacement")
        replacement.write_bytes(b"raise RuntimeError('path replacement must not execute')\n")
        os.replace(replacement, runtime["oracle_code"])

    oracle_completed = invoke_launcher(
        runtime,
        operation="ORACLE",
        request_path=oracle_request_path,
        state_root=state_root,
        mutate_after_open=replace_oracle_runtime_path,
    )
    assert oracle_completed.returncode == 0, oracle_completed.stdout + oracle_completed.stderr
    oracle_launch_receipt = json.loads(oracle_completed.stdout)
    assert oracle_launch_receipt["snapshot_mode"] == "SEALED_FD_COMPILE_EXEC_V2"
    assert oracle_launch_receipt["release_evidence_eligible"] is False
    assert oracle_launch_receipt["local_reproducible_only"] is True
    assert oracle_launch_receipt["outer_launch_provenance_status"] == (
        "CALLER_ASSERTED_PENDING_EXTERNAL_ANCHOR"
    )
    assert oracle_launch_receipt["runtime_environment_scope"] == (
        "HOST_LOCAL_NON_HERMETIC_STDLIB_NOT_BYTE_PINNED"
    )
    assert oracle_launch_receipt["caller_asserted_bootstrap_source_sha256"] == (
        launcher.PINNED_BOOTSTRAP_SOURCE_SHA256
    )
    assert oracle_launch_receipt["launcher_sha256"] == digest(runtime["launcher_bytes"])
    for field in (
        "bootstrap_attestation_sha256",
        "interpreter_executable_sha256",
        "interpreter_identity_sha256",
        "interpreter_flags_sha256",
        "sys_path_sha256",
    ):
        assert len(oracle_launch_receipt[field]) == 64
    oracle_manifest = json.loads(
        (
            launch_output_root(state_root, "ORACLE")
            / "oracle_output"
            / "oracle_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert oracle_manifest["release_evidence_eligible"] is False
    assert oracle_manifest["oracle_release_content_binding"] == {
        "code_sha256": launcher.PINNED_RELEASES["ORACLE"]["code_sha256"],
        "contract_sha256": launcher.PINNED_RELEASES["ORACLE"]["contract_sha256"],
        "schema_sha256": launcher.PINNED_RELEASES["ORACLE"]["schema_sha256"],
        "launcher_sha256": digest(runtime["launcher_bytes"]),
        "snapshot_mode": "SEALED_FD_COMPILE_EXEC_V2",
    }

    # Restore a byte-identical path only to obtain the next independent
    # inherited launcher FD.  The first process already proved that replacing
    # the path after the FD was opened did not change its executed bytes.
    Path(runtime["launcher"]).write_bytes(runtime["launcher_bytes"])
    verifier_request_path, oracle_snapshots = build_verifier_request(
        state_root, oracle_request, runtime
    )

    def replace_verifier_runtime_path() -> None:
        launcher_replacement = Path(runtime["launcher"]).with_suffix(".replacement")
        launcher_replacement.write_bytes(
            b"raise RuntimeError('verifier launcher path replacement executed')\n"
        )
        os.replace(launcher_replacement, runtime["launcher"])
        replacement = Path(runtime["verifier_code"]).with_suffix(".replacement")
        replacement.write_bytes(b"raise RuntimeError('verifier path replacement executed')\n")
        os.replace(replacement, runtime["verifier_code"])
        reference_code_replacement = Path(runtime["reference_code"]).with_suffix(
            ".replacement"
        )
        reference_code_replacement.write_bytes(
            b"raise RuntimeError('reference path replacement executed')\n"
        )
        os.replace(reference_code_replacement, runtime["reference_code"])
        reference_contract_replacement = Path(
            runtime["reference_contract"]
        ).with_suffix(".replacement")
        reference_contract_replacement.write_bytes(b"{}\n")
        os.replace(reference_contract_replacement, runtime["reference_contract"])

    verifier_completed = invoke_launcher(
        runtime,
        operation="VERIFIER",
        request_path=verifier_request_path,
        state_root=state_root,
        mutate_after_open=replace_verifier_runtime_path,
        overrides={
            "oracle_code": oracle_snapshots["oracle_code_snapshot"],
            "oracle_contract": oracle_snapshots["oracle_contract_snapshot"],
            "oracle_schema": oracle_snapshots["oracle_schema_snapshot"],
        },
    )
    assert verifier_completed.returncode == 0, (
        verifier_completed.stdout + verifier_completed.stderr
    )
    verifier_launch_receipt = json.loads(verifier_completed.stdout)
    assert verifier_launch_receipt["snapshot_mode"] == "SEALED_FD_COMPILE_EXEC_V2"
    assert verifier_launch_receipt["release_evidence_eligible"] is False
    assert verifier_launch_receipt["local_reproducible_only"] is True
    assert verifier_launch_receipt["outer_launch_provenance_status"] == (
        "CALLER_ASSERTED_PENDING_EXTERNAL_ANCHOR"
    )
    assert verifier_launch_receipt["runtime_environment_scope"] == (
        "HOST_LOCAL_NON_HERMETIC_STDLIB_NOT_BYTE_PINNED"
    )
    assert verifier_launch_receipt["caller_asserted_bootstrap_source_sha256"] == (
        launcher.PINNED_BOOTSTRAP_SOURCE_SHA256
    )
    assert len(verifier_launch_receipt["bootstrap_attestation_sha256"]) == 64
    assert verifier_launch_receipt["interpreter_executable_sha256"] == (
        oracle_launch_receipt["interpreter_executable_sha256"]
    )
    assert verifier_launch_receipt["sys_path_sha256"] == (
        oracle_launch_receipt["sys_path_sha256"]
    )
    verifier_receipt = json.loads(
        (
            launch_output_root(state_root, "VERIFIER")
            / "verifier_output"
            / "verifier_receipt.json"
        ).read_text(encoding="utf-8")
    )
    assert verifier_receipt["release_evidence_eligible"] is False
    assert verifier_receipt["oracle_release_content_binding"] == oracle_manifest[
        "oracle_release_content_binding"
    ]
    assert len(verifier_receipt["reference_result_sha256"]) == 64
    assert verifier_receipt["verifier_release_content_binding"] == {
        "code_sha256": launcher.PINNED_RELEASES["VERIFIER"]["code_sha256"],
        "schema_sha256": launcher.PINNED_RELEASES["VERIFIER"]["schema_sha256"],
        "reference_code_sha256": launcher.PINNED_RELEASES["REFERENCE"][
            "code_sha256"
        ],
        "reference_contract_sha256": launcher.PINNED_RELEASES["REFERENCE"][
            "contract_sha256"
        ],
        "reference_result_sha256": verifier_receipt["reference_result_sha256"],
        "launcher_sha256": digest(runtime["launcher_bytes"]),
        "snapshot_mode": "SEALED_FD_COMPILE_EXEC_V2",
    }
    verifier_commit_bytes = (
        launch_output_root(state_root, "VERIFIER")
        / "verifier_output"
        / "COMMIT.json"
    ).read_bytes()
    verifier_commit = json.loads(verifier_commit_bytes)
    assert set(verifier_commit) == {
        "schema_version",
        "request_sha256",
        "receipt_sha256",
        "receipt_size_bytes",
        "verifier_receipt_sha256",
    }
    verifier_receipt_bytes = (
        launch_output_root(state_root, "VERIFIER")
        / "verifier_output"
        / "verifier_receipt.json"
    ).read_bytes()
    assert verifier_commit["receipt_sha256"] == digest(verifier_receipt_bytes)
    assert verifier_commit["receipt_size_bytes"] == len(verifier_receipt_bytes)
    assert verifier_commit["verifier_receipt_sha256"] == verifier_receipt[
        "verifier_receipt_sha256"
    ]
    assert {
        path.relative_to(ROOT).as_posix(): (
            digest(path.read_bytes()),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for directory in ROOT.rglob("__pycache__")
        for path in directory.iterdir()
        if path.is_file()
    } == cache_snapshot_before


def test_launcher_surface_has_no_arbitrary_program_or_process_api() -> None:
    tree = ast.parse((ROOT / "paper_research_fd_launcher_v2.py").read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imports.isdisjoint({"socket", "subprocess", "requests", "urllib"})
    source = (ROOT / "paper_research_fd_launcher_v2.py").read_text(encoding="utf-8")
    assert "--executable" not in source
    assert "--argv" not in source
    assert "--reference-path" not in source
    assert "--reference-module" not in source
    assert "--reference-callable" not in source
    assert set(launcher.OPERATIONS) == {"ORACLE", "VERIFIER"}
