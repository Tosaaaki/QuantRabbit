from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from quant_rabbit.dojo_ai_discretion import canonical_sha256
from quant_rabbit.dojo_ai_validity import (
    DojoAIValidityError,
    append_artifact_commit,
    append_invalidation,
    assert_artifacts_valid,
    initialize_registry,
    status_artifact,
    verify_registry,
)


NOW = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)


def sealed(contract: str, key: str, **fields: object) -> dict:
    value = {"contract": contract, **fields}
    value[key] = canonical_sha256(value)
    return value


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def initialized(tmp_path: Path) -> tuple[Path, dict, dict]:
    run = tmp_path / "run"
    run.mkdir()
    precommit = sealed(
        "QR_DOJO_AI_FORWARD_PRECOMMIT_V3",
        "precommit_sha256",
        state="PRECOMMITTED",
    )
    start = sealed(
        "QR_DOJO_AI_FORWARD_START_V3",
        "start_receipt_sha256",
        state="STARTED",
        precommit_sha256=precommit["precommit_sha256"],
    )
    write(run / "precommit.json", precommit)
    write(run / "start.json", start)
    initialize_registry(
        run,
        precommit_path=run / "precommit.json",
        start_path=run / "start.json",
        created_at_utc=NOW,
    )
    return run, precommit, start


def test_first_write_bytes_and_transitive_invalidity_are_fail_closed(
    tmp_path: Path,
) -> None:
    run, precommit, start = initialized(tmp_path)
    day = sealed(
        "QR_DOJO_AI_DAY_SOURCE_SEAL_V3",
        "day_seal_sha256",
        precommit_sha256=precommit["precommit_sha256"],
        start_receipt_sha256=start["start_receipt_sha256"],
    )
    write(run / "days/day-001.json", day)
    append_artifact_commit(
        run,
        logical_id="day/001/seal",
        artifact_path=run / "days/day-001.json",
        parent_logical_ids=("precommit", "start"),
        committed_at_utc=NOW,
    )
    terminal = sealed(
        "QR_DOJO_AI_FORWARD_CELL_RESPONSE_V3",
        "cell_terminal_sha256",
        day_seal_sha256=day["day_seal_sha256"],
    )
    write(run / "responses/day-001/cell-a.json", terminal)
    append_artifact_commit(
        run,
        logical_id="day/001/cell/cell-a/terminal",
        artifact_path=run / "responses/day-001/cell-a.json",
        parent_logical_ids=("day/001/seal",),
        committed_at_utc=NOW,
    )
    snapshot = append_invalidation(
        run,
        logical_id="day/001/seal",
        reason_code="SOURCE_CAPTURE_CONTAMINATED",
        evidence_sha256=hashlib.sha256(b"evidence").hexdigest(),
        invalidated_at_utc=NOW,
    )
    assert snapshot.invalidated == {
        "day/001/seal",
        "day/001/cell/cell-a/terminal",
    }
    with pytest.raises(DojoAIValidityError, match="transitively invalid"):
        assert_artifacts_valid(run, ("day/001/cell/cell-a/terminal",))


def test_slot_reuse_and_byte_tamper_are_rejected(tmp_path: Path) -> None:
    run, precommit, start = initialized(tmp_path)
    day = sealed(
        "QR_DOJO_AI_DAY_SOURCE_SEAL_V3",
        "day_seal_sha256",
        precommit_sha256=precommit["precommit_sha256"],
        start_receipt_sha256=start["start_receipt_sha256"],
    )
    path = run / "days/day-001.json"
    write(path, day)
    append_artifact_commit(
        run,
        logical_id="day/001/seal",
        artifact_path=path,
        parent_logical_ids=("precommit", "start"),
        committed_at_utc=NOW,
    )
    replacement = sealed(
        "QR_DOJO_AI_DAY_SOURCE_SEAL_V3",
        "day_seal_sha256",
        precommit_sha256=precommit["precommit_sha256"],
        start_receipt_sha256=start["start_receipt_sha256"],
        changed=True,
    )
    write(run / "days/day-other.json", replacement)
    with pytest.raises(DojoAIValidityError, match="SLOT_REUSE_CONFLICT"):
        append_artifact_commit(
            run,
            logical_id="day/001/seal",
            artifact_path=run / "days/day-other.json",
            parent_logical_ids=("precommit", "start"),
            committed_at_utc=NOW,
        )
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(DojoAIValidityError, match="TAMPERED"):
        verify_registry(run)


def test_registry_never_claims_external_monotonicity_or_authority(
    tmp_path: Path,
) -> None:
    run, _, _ = initialized(tmp_path)
    status = status_artifact(run)
    assert status["external_witness_status"] == "ABSENT"
    assert "LOCAL_OWNER_CAN_DELETE_OR_RECREATE_ENTIRE_LEDGER" in status["limitations"]
    assert status["proof_eligible"] is False
    assert status["promotion_eligible"] is False
    assert status["live_permission"] is False


def test_ai_forward_cli_canonicalizes_relative_run_dir_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/run-dojo-ai-forward.py"
    module_spec = importlib.util.spec_from_file_location(
        "dojo_ai_forward_cli_test", script
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    observed: dict[str, Path] = {}

    def handler(args: object) -> dict[str, str]:
        observed["run_dir"] = args.run_dir
        return {"status": "OBSERVED"}

    monkeypatch.setattr(module, "_validity_status", handler)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "validity-status", "--run-dir", "relative-run"],
    )

    assert module.main() == 0
    assert observed["run_dir"] == (tmp_path / "relative-run").resolve()
    assert json.loads(capsys.readouterr().out)["status"] == "OBSERVED"
