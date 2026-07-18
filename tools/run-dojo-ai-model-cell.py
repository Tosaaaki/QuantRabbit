#!/usr/bin/env python3
"""Execute exactly one DOJO AI cell through an isolated ChatGPT-auth Codex CLI.

Run this operator tool from an external terminal or scheduler, never from inside
another Codex turn.  QuantRabbit production code is not imported into a model
API path; this tool launches a fresh CLI process, persists launch intent before
Popen, captures the first raw JSONL stream, and permanently seals success or
failure without retry or response selection.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_rabbit.dojo_ai_execution import (  # noqa: E402
    DojoAIExecutionError,
    build_execution_receipt,
    build_execution_request,
    validate_execution_receipt,
    validate_execution_request,
)
from quant_rabbit.dojo_ai_forward import (  # noqa: E402
    build_executed_cell_terminal,
    validate_cell_terminal,
    validate_day_seal,
    validate_precommit,
    validate_start_receipt,
    verify_source_bindings_against_repo,
)
from quant_rabbit.dojo_prompt_phase import assert_locked_preregistration  # noqa: E402
from quant_rabbit.dojo_ai_validity import (  # noqa: E402
    append_artifact_commit,
    assert_artifacts_valid,
)


DEFAULT_REGISTRY = REPO / "research/registries/dojo_prompt_experiment_v1.json"
DEFAULT_CODEX = Path("/Applications/ChatGPT.app/Contents/Resources/codex")
DEFAULT_CODEX_HOME = Path("/Users/tossaki/.codex")
_PARENT_CONTEXT_MARKERS = (
    "CODEX_THREAD_ID",
    "CODEX_INTERNAL_ORIGINATOR_OVERRIDE",
)
_FORBIDDEN_PROVIDER_ENV = (
    "OPENAI_API_KEY",
    "QR_OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)
_FIXED_PATH = "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/Applications/ChatGPT.app/Contents/Resources"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _terminal_path(run_dir: Path, ordinal: int, cell_id: str) -> Path:
    return run_dir / "responses" / f"day-{ordinal:03d}" / f"{cell_id}.json"


def _execution_dir(run_dir: Path, ordinal: int, cell_id: str) -> Path:
    return run_dir / "model-executions" / f"day-{ordinal:03d}" / cell_id


def _load_context(
    run_dir: Path,
    registry_path: Path,
    ordinal: int,
    cell_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    registry = assert_locked_preregistration(_strict_json(registry_path))
    precommit = validate_precommit(_strict_json(run_dir / "precommit.json"))
    verify_source_bindings_against_repo(precommit["source_bindings"], REPO)
    start = validate_start_receipt(_strict_json(run_dir / "start.json"), precommit)
    assert_artifacts_valid(run_dir, ("precommit", "start"))
    previous: dict[str, Any] | None = None
    days: list[dict[str, Any]] = []
    day_dir = run_dir / "days"
    for expected in range(1, ordinal + 1):
        path = day_dir / f"day-{expected:03d}.json"
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("AI execution day chain is incomplete or unsafe")
        day = validate_day_seal(
            _strict_json(path),
            registry,
            precommit,
            start,
            previous,
            expected_ordinal=expected,
        )
        days.append(day)
        previous = day
    day = days[-1]
    assert_artifacts_valid(
        run_dir,
        tuple(f"day/{index:03d}/seal" for index in range(1, ordinal + 1)),
    )
    if day["state"] != "REQUESTS_SEALED":
        raise RuntimeError("AI execution requires a request-sealed day")
    matches = [
        cell
        for cell in day["cells"]
        if cell.get("request_receipt", {}).get("cell_id") == cell_id
    ]
    if len(matches) != 1:
        raise RuntimeError("AI execution cell is not unique in its sealed day")
    return precommit, start, days[-2] if len(days) > 1 else None, day, matches[0]


def _assert_next_cell(
    run_dir: Path,
    registry: Mapping[str, Any],
    precommit: Mapping[str, Any],
    start: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    day: Mapping[str, Any],
    cell_id: str,
) -> dict[str, Any] | None:
    ordered = [row["cell_id"] for row in day["schedule"]["cells"]]
    target = ordered.index(cell_id)
    for prior_id in ordered[:target]:
        prior_path = _terminal_path(run_dir, day["ordinal"], prior_id)
        if not prior_path.is_file() or prior_path.is_symlink():
            raise RuntimeError("AI cells must execute in the preregistered order")
        validate_cell_terminal(
            _strict_json(prior_path), registry, precommit, start, previous, day
        )
        assert_artifacts_valid(
            run_dir,
            (f"day/{day['ordinal']:03d}/cell/{prior_id}/terminal",),
        )
    path = _terminal_path(run_dir, day["ordinal"], cell_id)
    if path.is_file():
        terminal = validate_cell_terminal(
            _strict_json(path), registry, precommit, start, previous, day
        )
        if terminal["state"] not in {
            "EXECUTED_RESPONSE_SEALED",
            "MODEL_EXECUTION_FAILED",
        }:
            raise RuntimeError("cell already contains non-execution diagnostic material")
        assert_artifacts_valid(
            run_dir,
            (f"day/{day['ordinal']:03d}/cell/{cell_id}/terminal",),
        )
        return terminal
    if path.is_symlink():
        raise RuntimeError("AI execution terminal path is an unsafe symlink")
    return None


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    _assert_external_boundary()
    args.run_dir = args.run_dir.resolve()
    with _exclusive_lock(args.run_dir / ".ai-model-executor.lock"):
        registry = assert_locked_preregistration(_strict_json(args.registry))
        precommit, start, previous, day, cell = _load_context(
            args.run_dir, args.registry, args.ordinal, args.cell_id
        )
        existing = _assert_next_cell(
            args.run_dir,
            registry,
            precommit,
            start,
            previous,
            day,
            args.cell_id,
        )
        if existing is not None:
            return existing
        _assert_truth_absent(args.run_dir, args.ordinal)
        evidence_dir = _execution_dir(args.run_dir, args.ordinal, args.cell_id)
        request_path = evidence_dir / "invocation-request.json"
        receipt_path = evidence_dir / "execution-receipt.json"
        terminal_path = _terminal_path(args.run_dir, args.ordinal, args.cell_id)
        if receipt_path.is_file():
            receipt = validate_execution_receipt(_strict_json(receipt_path))
            _commit_execution_request(args.run_dir, args.ordinal, args.cell_id)
            _commit_execution_receipt(args.run_dir, args.ordinal, args.cell_id)
            terminal = build_executed_cell_terminal(
                registry, precommit, start, previous, day, receipt
            )
            _write_json_new_or_same(terminal_path, terminal, args.run_dir)
            _commit_execution_terminal(args.run_dir, args.ordinal, args.cell_id)
            return terminal
        if request_path.is_file():
            request = validate_execution_request(_strict_json(request_path))
            _commit_execution_request(args.run_dir, args.ordinal, args.cell_id)
            created = _parse_utc(request["created_at_utc"])
            receipt = build_execution_receipt(
                request,
                raw_stdout_jsonl=b"",
                raw_stderr=b"launch intent exists without a sealed first transcript",
                output_last_message=b"",
                returncode=None,
                timed_out=False,
                started_at_utc=created,
                completed_at_utc=created,
                launch_started=False,
            )
            _write_json_new_or_same(receipt_path, receipt, args.run_dir)
            _commit_execution_receipt(args.run_dir, args.ordinal, args.cell_id)
            terminal = build_executed_cell_terminal(
                registry, precommit, start, previous, day, receipt
            )
            _write_json_new_or_same(terminal_path, terminal, args.run_dir)
            _commit_execution_terminal(args.run_dir, args.ordinal, args.cell_id)
            return terminal
        if request_path.is_symlink() or receipt_path.is_symlink():
            raise RuntimeError("AI execution evidence path is an unsafe symlink")

        binary = args.codex_binary.resolve(strict=True)
        binary_sha = _file_sha256(binary)
        with tempfile.TemporaryDirectory(prefix="qr-dojo-ai-") as raw_root:
            runtime_root = Path(raw_root)
            working = runtime_root / "empty-workspace"
            task_home = runtime_root / "home"
            task_codex_home = runtime_root / "codex-home"
            working.mkdir(mode=0o700)
            task_home.mkdir(mode=0o700)
            task_codex_home.mkdir(mode=0o700)
            auth_source = args.codex_home.resolve(strict=True) / "auth.json"
            if not auth_source.is_file() or auth_source.is_symlink():
                raise RuntimeError("ChatGPT auth.json is absent or unsafe")
            os.symlink(auth_source, task_codex_home / "auth.json")
            probe_env = _clean_env(
                code_home=args.codex_home.resolve(strict=True),
                home=task_home,
                tmpdir=runtime_root,
            )
            version = _run_probe([str(binary), "--version"], probe_env)
            auth_status = _run_probe([str(binary), "login", "status"], probe_env)
            if auth_status != "Logged in using ChatGPT":
                raise RuntimeError("Codex CLI is not using ChatGPT authentication")
            auth_probe_sha = hashlib.sha256(auth_status.encode("utf-8")).hexdigest()
            runtime_identity = hashlib.sha256(
                str(runtime_root).encode("utf-8")
            ).hexdigest()
            execution_cell = {
                **cell,
                "prompt_lock": precommit["prompt_locks"][cell["variant_id"]],
            }
            request = build_execution_request(
                precommit_sha256=precommit["precommit_sha256"],
                day_seal_sha256=day["day_seal_sha256"],
                cell=execution_cell,
                cli_binary_path=str(binary),
                cli_binary_sha256=binary_sha,
                cli_version=version,
                auth_mode_probe_sha256=auth_probe_sha,
                runtime_root_identity_sha256=runtime_identity,
                created_at_utc=_now(),
            )
            _write_json_new_or_same(request_path, request, args.run_dir)
            _commit_execution_request(args.run_dir, args.ordinal, args.cell_id)
            schema_path = runtime_root / "output-schema.json"
            last_message_path = runtime_root / "last-message.json"
            stdout_path = evidence_dir / "first-stdout.jsonl"
            stderr_path = evidence_dir / "first-stderr.txt"
            _write_bytes_new(schema_path, _canonical_bytes(request["output_schema"]))
            env = _clean_env(
                code_home=task_codex_home,
                home=task_home,
                tmpdir=runtime_root,
            )
            command = _command(
                binary,
                working,
                schema_path,
                last_message_path,
                request,
            )
            started = _now()
            timed_out = False
            returncode: int | None = None
            evidence_dir.mkdir(parents=True, exist_ok=True)
            with _new_binary_writer(stdout_path) as stdout_file, _new_binary_writer(
                stderr_path
            ) as stderr_file:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    env=env,
                    cwd=working,
                    start_new_session=True,
                )
                try:
                    process.communicate(
                        request["stdin_envelope"].encode("utf-8"),
                        timeout=args.timeout_seconds,
                    )
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                returncode = process.returncode
                stdout_file.flush()
                stderr_file.flush()
                os.fsync(stdout_file.fileno())
                os.fsync(stderr_file.fileno())
            completed = _now()
            raw_stdout = stdout_path.read_bytes()
            raw_stderr = stderr_path.read_bytes()
            output = (
                last_message_path.read_bytes()
                if last_message_path.is_file() and not last_message_path.is_symlink()
                else b""
            )
            try:
                receipt = build_execution_receipt(
                    request,
                    raw_stdout_jsonl=raw_stdout,
                    raw_stderr=raw_stderr,
                    output_last_message=output,
                    returncode=returncode,
                    timed_out=timed_out,
                    started_at_utc=started,
                    completed_at_utc=completed,
                )
            except DojoAIExecutionError:
                # Intent and raw first stream are already immutable.  A later
                # invocation will seal deterministic failure without relaunch.
                raise
            _write_json_new_or_same(receipt_path, receipt, args.run_dir)
            _commit_execution_receipt(args.run_dir, args.ordinal, args.cell_id)
            terminal = build_executed_cell_terminal(
                registry, precommit, start, previous, day, receipt
            )
            _write_json_new_or_same(terminal_path, terminal, args.run_dir)
            _commit_execution_terminal(args.run_dir, args.ordinal, args.cell_id)
            return terminal


def _execution_logical(ordinal: int, cell_id: str, stage: str) -> str:
    return f"day/{ordinal:03d}/cell/{cell_id}/{stage}"


def _commit_execution_request(run_dir: Path, ordinal: int, cell_id: str) -> None:
    append_artifact_commit(
        run_dir,
        logical_id=_execution_logical(ordinal, cell_id, "execution-request"),
        artifact_path=_execution_dir(run_dir, ordinal, cell_id)
        / "invocation-request.json",
        parent_logical_ids=("precommit", f"day/{ordinal:03d}/seal"),
        committed_at_utc=_now(),
    )


def _commit_execution_receipt(run_dir: Path, ordinal: int, cell_id: str) -> None:
    append_artifact_commit(
        run_dir,
        logical_id=_execution_logical(ordinal, cell_id, "execution-receipt"),
        artifact_path=_execution_dir(run_dir, ordinal, cell_id)
        / "execution-receipt.json",
        parent_logical_ids=(
            _execution_logical(ordinal, cell_id, "execution-request"),
        ),
        committed_at_utc=_now(),
    )


def _commit_execution_terminal(run_dir: Path, ordinal: int, cell_id: str) -> None:
    append_artifact_commit(
        run_dir,
        logical_id=_execution_logical(ordinal, cell_id, "terminal"),
        artifact_path=_terminal_path(run_dir, ordinal, cell_id),
        parent_logical_ids=(
            f"day/{ordinal:03d}/seal",
            _execution_logical(ordinal, cell_id, "execution-receipt"),
        ),
        committed_at_utc=_now(),
    )


def _command(
    binary: Path,
    working: Path,
    schema_path: Path,
    last_message_path: Path,
    request: Mapping[str, Any],
) -> list[str]:
    runtime = request["runtime"]
    command = [
        str(binary),
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
        "--skip-git-repo-check",
        "--json",
        "--color",
        "never",
        "--sandbox",
        "read-only",
        "--cd",
        str(working),
        "--model",
        str(request["requested_model"]),
        "--config",
        f'model_reasoning_effort="{request["reasoning_effort"]}"',
        "--config",
        'web_search="disabled"',
    ]
    for feature in runtime["disabled_features"]:
        command.extend(("--disable", str(feature)))
    command.extend(
        (
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(last_message_path),
            "-",
        )
    )
    return command


def _assert_external_boundary() -> None:
    present = [name for name in _PARENT_CONTEXT_MARKERS if name in os.environ]
    forbidden = [name for name in _FORBIDDEN_PROVIDER_ENV if name in os.environ]
    if present:
        raise RuntimeError(
            "AI model cells must launch from an external non-Codex parent: "
            + ",".join(present)
        )
    if forbidden:
        raise RuntimeError(
            "AI model cells refuse ambient provider/API key variables: "
            + ",".join(forbidden)
        )


def _assert_truth_absent(run_dir: Path, ordinal: int) -> None:
    forbidden = (
        run_dir / "truth" / f"day-{ordinal:03d}",
        run_dir / "phase-score.json",
    )
    if any(path.exists() or path.is_symlink() for path in forbidden):
        raise RuntimeError("truth or phase material exists before model execution")


def _clean_env(*, code_home: Path, home: Path, tmpdir: Path) -> dict[str, str]:
    return {
        "CODEX_HOME": str(code_home),
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": _FIXED_PATH,
        "TMPDIR": str(tmpdir),
    }


def _run_probe(command: list[str], env: Mapping[str, str]) -> str:
    result = subprocess.run(
        command,
        env=dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=10,
    )
    text = result.stdout.decode("utf-8", errors="strict").strip()
    if result.returncode != 0 or not text or len(text) > 1_000:
        raise RuntimeError("Codex CLI preflight failed")
    return text


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError("AI model executor lock is an unsafe symlink")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def _new_binary_writer(path: Path) -> Iterator[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    stream = os.fdopen(descriptor, "wb")
    try:
        yield stream
    finally:
        stream.close()
        _fsync_directory(path.parent)


def _write_bytes_new(path: Path, payload: bytes) -> None:
    with _new_binary_writer(path) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_new_or_same(path: Path, value: Mapping[str, Any], root: Path) -> None:
    root = root.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.resolve(strict=False) == root or root not in path.resolve(strict=False).parents:
        raise RuntimeError("AI execution write escaped its run directory")
    payload = _canonical_bytes(value) + b"\n"
    if path.is_file() and not path.is_symlink():
        if path.read_bytes() != payload:
            raise RuntimeError("immutable AI execution artifact already differs")
        return
    if path.exists() or path.is_symlink():
        raise RuntimeError("AI execution artifact target is unsafe")
    _write_bytes_new(path, payload)


def _strict_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"AI execution JSON path is absent or unsafe: {path}")

    def reject_constant(token: str) -> None:
        raise RuntimeError(f"non-finite JSON number: {token}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, dict):
        raise RuntimeError("AI execution JSON artifact must be an object")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | (os.O_DIRECTORY if hasattr(os, "O_DIRECTORY") else 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ordinal", type=int, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--codex-binary", type=Path, default=DEFAULT_CODEX)
    parser.add_argument("--codex-home", type=Path, default=DEFAULT_CODEX_HOME)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.ordinal < 1 or not 30 <= args.timeout_seconds <= 1_800:
        parser.error("ordinal or timeout is outside the fixed safe bounds")
    result = _execute(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
