"""Independent black-box adversarial checks for the file-only shadow CLI.

This module intentionally does not import the runtime, its accounting module, or
the evidence builder.  It treats the CLI and its durable files as the public
boundary and independently tampers with copies of completed state.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "forward_shadow_core_v2.py"
PYTHON = Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _embedded_hash(payload: dict, field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def _records(count: int = 4) -> list[dict]:
    start = 1_781_510_400_000_000_000
    result = []
    for index in range(count):
        source = start + index * 1_000_000_000
        result.append(
            {
                "schema_version": 1,
                "provider_id": "INDEPENDENT_FILE_FIXTURE",
                "instrument": "EUR_USD",
                "bid": f"1.{10000 + index:05d}",
                "ask": f"1.{10120 + index:05d}",
                "liquidity_optional": "1000000",
                "source_ts_ns": source,
                "arrival_ts_ns": source + 50_000_000,
                "provider_event_id": f"BLACKBOX-{index + 1}",
                "sequence": index + 1,
                "heartbeat": False,
                "quality_flags": [],
            }
        )
    return result


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_canonical(row) + b"\n" for row in rows))
    return path


def _cli(*arguments: object, expect_ok: bool) -> tuple[subprocess.CompletedProcess, dict]:
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
    }
    completed = subprocess.run(
        [str(PYTHON), str(RUNTIME), *(str(item) for item in arguments)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["ok"] is expect_ok
    assert (completed.returncode == 0) is expect_ok
    return completed, payload


def _completed_state(tmp_path: Path) -> tuple[Path, Path, dict]:
    source = _write_jsonl(tmp_path / "events.jsonl", _records())
    state = tmp_path / "state"
    _, receipt = _cli("ingest-batch", source, "--state-dir", state, expect_ok=True)
    return source, state, receipt["result"]


@pytest.mark.parametrize(
    ("mutation", "expected_codes"),
    [
        ("manifest_only", {"MANIFEST_LEDGER_BINDING_MISMATCH"}),
        ("ledger_only", {"MANIFEST_LEDGER_BINDING_MISMATCH"}),
        ("checkpoint_only", {"CHECKPOINT_AHEAD_OF_LEDGER", "CHECKPOINT_MISMATCH"}),
        ("manifest_missing", {"MANIFEST_LEDGER_BINDING_MISMATCH"}),
        (
            "raw_ledger_missing",
            {"MANIFEST_LEDGER_BINDING_MISMATCH", "CHECKPOINT_AHEAD_OF_LEDGER"},
        ),
        ("checkpoint_missing", {"CHECKPOINT_MISSING_FOR_EXISTING_STATE"}),
        ("semantic_reseal", {"SOURCE_MANIFEST_COUNT_MISMATCH"}),
        ("orphan_blob", {"SOURCE_BLOB_BINDING_MISMATCH"}),
    ],
)
def test_black_box_restart_cross_binding_fails_closed(
    tmp_path: Path, mutation: str, expected_codes: set[str]
) -> None:
    _, state, _ = _completed_state(tmp_path / "original")
    clone = tmp_path / mutation
    shutil.copytree(state, clone)
    if mutation == "manifest_only":
        for name in (
            "raw_bbo_ledger.jsonl",
            "proposal_stream_ledger.jsonl",
            "virtual_execution_ledger.jsonl",
            "restart_checkpoint.json",
        ):
            (clone / name).unlink(missing_ok=True)
    elif mutation == "ledger_only":
        shutil.rmtree(clone / "batch_manifests")
        shutil.rmtree(clone / "source_blobs")
    elif mutation == "checkpoint_only":
        shutil.rmtree(clone / "batch_manifests")
        shutil.rmtree(clone / "source_blobs")
        for name in (
            "raw_bbo_ledger.jsonl",
            "proposal_stream_ledger.jsonl",
            "virtual_execution_ledger.jsonl",
        ):
            (clone / name).unlink(missing_ok=True)
    elif mutation == "manifest_missing":
        next((clone / "batch_manifests").glob("*.json")).unlink()
    elif mutation == "raw_ledger_missing":
        (clone / "raw_bbo_ledger.jsonl").unlink()
    elif mutation == "checkpoint_missing":
        (clone / "restart_checkpoint.json").unlink()
    elif mutation == "semantic_reseal":
        manifest_path = next((clone / "batch_manifests").glob("*.json"))
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["event_count"] = 999
        payload["manifest_sha256"] = _embedded_hash(payload, "manifest_sha256")
        manifest_path.write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif mutation == "orphan_blob":
        (clone / "source_blobs" / f"{'f' * 64}.blob").write_bytes(b"orphan")
    _, failure = _cli("status", "--state-dir", clone, expect_ok=False)
    assert failure["error_code"] in expected_codes


def test_black_box_clean_environment_mtime_idempotence_and_symlink_rejection(
    tmp_path: Path,
) -> None:
    source, state, first = _completed_state(tmp_path)
    assert first["idempotent_reingest"] is False
    before = source.stat().st_mtime_ns
    os.utime(source, ns=(before + 10_000_000, before + 10_000_000))
    _, second = _cli("ingest-batch", source, "--state-dir", state, expect_ok=True)
    assert second["result"]["idempotent_reingest"] is True

    source_link = tmp_path / "events-link.jsonl"
    source_link.symlink_to(source)
    linked_state = tmp_path / "linked-source-state"
    _, rejected = _cli(
        "ingest-batch", source_link, "--state-dir", linked_state, expect_ok=False
    )
    assert rejected["error_code"] == "SYMLINK_FORBIDDEN"
    _, halted = _cli("status", "--state-dir", linked_state, expect_ok=True)
    assert halted["result"]["halt_new_actions"] is True

    external = tmp_path / "external-ledger.jsonl"
    shutil.copy2(state / "raw_bbo_ledger.jsonl", external)
    (state / "raw_bbo_ledger.jsonl").unlink()
    (state / "raw_bbo_ledger.jsonl").symlink_to(external)
    _, rejected_state = _cli("status", "--state-dir", state, expect_ok=False)
    assert rejected_state["error_code"] == "SYMLINK_FORBIDDEN"


@pytest.mark.parametrize(
    "payload",
    [b"", b'{"schema_version":1', b"\xff\xfe\n"],
    ids=["empty", "truncated", "invalid-utf8"],
)
def test_black_box_invalid_input_creates_a_durable_permanent_halt(
    tmp_path: Path, payload: bytes
) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_bytes(payload)
    state = tmp_path / "state"
    _, failure = _cli("ingest-batch", source, "--state-dir", state, expect_ok=False)
    assert failure["error_code"] in {
        "EMPTY_SOURCE_BATCH",
        "TRUNCATED_SOURCE_RECORD",
        "INVALID_UTF8_RECORD",
    }
    _, status = _cli("status", "--state-dir", state, expect_ok=True)
    assert status["result"]["halt_new_actions"] is True
    manifests = list((state / "batch_manifests").glob("*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED"
    assert manifest["failure_code"] == failure["error_code"]
    assert manifest["external_order_count"] == 0
