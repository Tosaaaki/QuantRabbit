from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from quant_rabbit.dojo_ai_tuning_state import verify_state_store
from quant_rabbit.dojo_candidate_lineage_registry import bind_result
from tests.test_dojo_ai_tuning_state import (
    _baseline,
    _evaluation,
    _raw_proposal,
    _seal_pending_second,
    _study,
    _submission,
    _write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run-dojo-ai-tuning-state.py"


def _run(*args: object) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(arg) for arg in args)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _accepted(result: subprocess.CompletedProcess[str]) -> dict:
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    value = json.loads(result.stdout)
    assert (
        result.stdout
        == json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    assert value["live_permission"] is False
    assert value["order_authority"] == "NONE"
    assert value["broker_mutation_allowed"] is False
    return value


def _rejected(result: subprocess.CompletedProcess[str]) -> dict:
    assert result.returncode == 2
    assert result.stdout == ""
    value = json.loads(result.stderr)
    assert value["status"] == "REJECTED"
    assert value["live_permission"] is False
    assert value["order_authority"] == "NONE"
    assert value["broker_mutation_allowed"] is False
    return value


def _cas(value: dict) -> tuple[str, str]:
    store = value["state_store"]
    return store["latest_event_sha256"], store["latest_state_sha256"]


def _init_cli(root: Path) -> tuple[Path, object, dict, Path, dict]:
    events, lineage, baseline, _ = _baseline(root)
    store = root.parent / f"{root.name}-state"
    study_path = root / "artifacts" / "study-1.json"
    initialized = _accepted(
        _run(
            "init",
            "--state-events",
            store,
            "--lineage-events",
            events,
            "--artifact-root",
            root,
            "--sealed-study",
            study_path,
            "--expected-lineage-tip-sha256",
            lineage.latest_event_sha256,
        )
    )
    assert initialized["state_store"]["event_count"] == 1
    assert initialized["state_store"]["automation_ready"] is True
    assert initialized["tuning_status"]["phase"] == "READY_FOR_MODEL"
    return events, lineage, baseline, store, initialized


def test_cli_full_valid_transition_path_is_cas_bound_and_research_only(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lineage-root"
    events, lineage, _, store, current = _init_cli(root)
    request = root / "artifacts" / "request-2.json"
    request.write_text('{"prompt":"bounded TRAIN proposal"}\n', encoding="utf-8")
    tip, parent = _cas(current)
    current = _accepted(
        _run(
            "reserve-model",
            "--state-events",
            store,
            "--expected-tip-event-sha256",
            tip,
            "--expected-parent-state-sha256",
            parent,
            "--lineage-events",
            events,
            "--artifact-root",
            root,
            "--invocation-id",
            "cli-model-a2",
            "--request-artifact",
            request,
            "--event-at-utc",
            "2026-07-20T00:01:00Z",
        )
    )
    assert current["tuning_status"]["phase"] == "MODEL_INVOCATION_RESERVED"

    proposal = _raw_proposal("qr-a2-cli", 2)
    raw_response = json.dumps(
        [_submission("cli-submission-a2", proposal)],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    response_input = root / "model-output.tmp"
    response_input.write_bytes(raw_response)
    response_dir = root / "model-responses"
    response_dir.mkdir()
    response_artifact = response_dir / "cli-model-a2.response.json"
    tip, parent = _cas(current)
    current = _accepted(
        _run(
            "record-response",
            "--state-events",
            store,
            "--expected-tip-event-sha256",
            tip,
            "--expected-parent-state-sha256",
            parent,
            "--artifact-root",
            root,
            "--invocation-id",
            "cli-model-a2",
            "--response-input",
            response_input,
            "--response-artifact",
            response_artifact,
            "--event-at-utc",
            "2026-07-20T00:02:00Z",
        )
    )
    assert response_artifact.read_bytes() == raw_response
    assert current["artifact_receipt"] == {
        "created_exclusively": True,
        "path": str(response_artifact.absolute()),
        "sha256": hashlib.sha256(raw_response).hexdigest(),
        "size_bytes": len(raw_response),
        "strict_json": True,
    }
    assert current["tuning_status"]["current_accepted_candidate_ids"] == ["qr-a2-cli"]

    latest = verify_state_store(store)["latest_state"]
    sealed_proposal = latest["attempts"][-1]["invocations"][0]["submissions"][0][
        "sealed_proposal"
    ]
    second = _study(2, [sealed_proposal])
    pending = _seal_pending_second(root, events, lineage, second)
    second_path = root / "artifacts" / "study-2.json"
    tip, parent = _cas(current)
    current = _accepted(
        _run(
            "reserve-run",
            "--state-events",
            store,
            "--expected-tip-event-sha256",
            tip,
            "--expected-parent-state-sha256",
            parent,
            "--lineage-events",
            events,
            "--artifact-root",
            root,
            "--sealed-study",
            second_path,
            "--dispatch-id",
            "cli-run-a2",
            "--event-at-utc",
            "2026-07-20T00:03:00Z",
        )
    )
    assert current["tuning_status"]["phase"] == "RUN_DISPATCH_RESERVED"

    tip, parent = _cas(current)
    current = _accepted(
        _run(
            "mark-dispatched",
            "--state-events",
            store,
            "--expected-tip-event-sha256",
            tip,
            "--expected-parent-state-sha256",
            parent,
            "--lineage-events",
            events,
            "--artifact-root",
            root,
            "--dispatch-id",
            "cli-run-a2",
            "--event-at-utc",
            "2026-07-20T00:04:00Z",
        )
    )
    assert current["tuning_status"]["phase"] == "AWAITING_LINEAGE_RESULT"

    evaluation_path = _write_json(
        root, "artifacts/evaluation-2.json", _evaluation(second)
    )
    complete = bind_result(
        events,
        artifact_root=root,
        evaluation_path=evaluation_path,
        expected_tip_sha256=pending.latest_event_sha256,
        event_at_utc="2026-07-20T00:05:00Z",
    )
    tip, parent = _cas(current)
    current = _accepted(
        _run(
            "bind-result",
            "--state-events",
            store,
            "--expected-tip-event-sha256",
            tip,
            "--expected-parent-state-sha256",
            parent,
            "--lineage-events",
            events,
            "--artifact-root",
            root,
            "--event-at-utc",
            "2026-07-20T00:06:00Z",
        )
    )
    assert current["tuning_status"]["phase"] == "READY_FOR_MODEL"
    assert current["tuning_status"]["attempts_consumed"] == 2
    assert (
        verify_state_store(store)["latest_state"]["last_terminal_result_binding"][
            "lineage_tip_sha256"
        ]
        == complete.latest_event_sha256
    )


def test_invalid_response_is_saved_first_charged_and_exact_retry_is_idempotent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "invalid-root"
    events, _, _, store, current = _init_cli(root)
    request = root / "artifacts" / "request.json"
    request.write_text("{}\n", encoding="utf-8")
    tip, parent = _cas(current)
    current = _accepted(
        _run(
            "reserve-model",
            "--state-events",
            store,
            "--expected-tip-event-sha256",
            tip,
            "--expected-parent-state-sha256",
            parent,
            "--lineage-events",
            events,
            "--artifact-root",
            root,
            "--invocation-id",
            "invalid-model-a2",
            "--request-artifact",
            request,
            "--event-at-utc",
            "2026-07-20T00:01:00Z",
        )
    )
    response_input = root / "invalid-model-output.tmp"
    # An empty model response is still a paid model call and must consume one
    # proposal slot after its exact zero-byte artifact is durably preserved.
    response_input.write_bytes(b"")
    response_dir = root / "model-responses"
    response_dir.mkdir()
    response_artifact = response_dir / "invalid.response"
    old_tip, old_parent = _cas(current)
    args = (
        "record-response",
        "--state-events",
        store,
        "--expected-tip-event-sha256",
        old_tip,
        "--expected-parent-state-sha256",
        old_parent,
        "--artifact-root",
        root,
        "--invocation-id",
        "invalid-model-a2",
        "--response-input",
        response_input,
        "--response-artifact",
        response_artifact,
        "--event-at-utc",
        "2026-07-20T00:02:00Z",
    )
    recorded = _accepted(_run(*args))
    assert recorded["artifact_receipt"]["strict_json"] is False
    assert recorded["tuning_status"]["invalid_proposal_count"] == 1
    assert recorded["tuning_status"]["proposal_slots_consumed"] == 2
    assert recorded["state_store"]["event_count"] == 3
    state = verify_state_store(store)["latest_state"]
    invocation = state["attempts"][-1]["invocations"][0]
    assert (
        invocation["response_sha256"]
        == hashlib.sha256(response_artifact.read_bytes()).hexdigest()
    )

    retried = _accepted(_run(*args))
    assert retried["state_store"]["event_count"] == 3
    assert retried["artifact_receipt"]["created_exclusively"] is False

    response_input.write_bytes(b"different invalid output")
    rejected = _rejected(_run(*args))
    assert "different bytes" in rejected["error"]
    assert response_artifact.read_bytes() == b""
    assert verify_state_store(store)["event_count"] == 3


def test_request_json_cas_and_incomplete_review_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "review-root"
    events, _, _, store, current = _init_cli(root)
    duplicate_json = root / "artifacts" / "duplicate-request.json"
    duplicate_json.write_text('{"prompt":1,"prompt":2}\n', encoding="utf-8")
    tip, parent = _cas(current)
    rejected = _rejected(
        _run(
            "reserve-model",
            "--state-events",
            store,
            "--expected-tip-event-sha256",
            tip,
            "--expected-parent-state-sha256",
            parent,
            "--lineage-events",
            events,
            "--artifact-root",
            root,
            "--invocation-id",
            "bad-request",
            "--request-artifact",
            duplicate_json,
            "--event-at-utc",
            "2026-07-20T00:01:00Z",
        )
    )
    assert "duplicate key" in rejected["error"]
    assert verify_state_store(store)["event_count"] == 1

    request = root / "artifacts" / "request.json"
    request.write_text("{}\n", encoding="utf-8")
    stale = _rejected(
        _run(
            "reserve-model",
            "--state-events",
            store,
            "--expected-tip-event-sha256",
            "0" * 64,
            "--expected-parent-state-sha256",
            parent,
            "--lineage-events",
            events,
            "--artifact-root",
            root,
            "--invocation-id",
            "stale-request",
            "--request-artifact",
            request,
            "--event-at-utc",
            "2026-07-20T00:01:00Z",
        )
    )
    assert "stale or forked state-store event tip" in stale["error"]
    assert verify_state_store(store)["event_count"] == 1

    current = _accepted(
        _run(
            "reserve-model",
            "--state-events",
            store,
            "--expected-tip-event-sha256",
            tip,
            "--expected-parent-state-sha256",
            parent,
            "--lineage-events",
            events,
            "--artifact-root",
            root,
            "--invocation-id",
            "crashed-model",
            "--request-artifact",
            request,
            "--event-at-utc",
            "2026-07-20T00:01:00Z",
        )
    )
    tip, parent = _cas(current)
    current = _accepted(
        _run(
            "mark-incomplete",
            "--state-events",
            store,
            "--expected-tip-event-sha256",
            tip,
            "--expected-parent-state-sha256",
            parent,
            "--reason-code",
            "MODEL_RESPONSE_LOST_AFTER_RESERVATION",
            "--event-at-utc",
            "2026-07-20T00:02:00Z",
        )
    )
    assert current["tuning_status"]["phase"] == "REVIEW_REQUIRED"
    assert current["tuning_status"]["attempts_consumed"] == 2

    tip, parent = _cas(current)
    current = _accepted(
        _run(
            "abandon",
            "--state-events",
            store,
            "--expected-tip-event-sha256",
            tip,
            "--expected-parent-state-sha256",
            parent,
            "--review-id",
            "operator-review-cli",
            "--rationale",
            "No model response exists; consume the attempt and terminate.",
            "--event-at-utc",
            "2026-07-20T00:03:00Z",
        )
    )
    assert current["tuning_status"]["phase"] == "TERMINATED"
    assert current["tuning_status"]["live_permission"] is False


def test_init_rejects_stale_lineage_tip_and_response_symlink(tmp_path: Path) -> None:
    root = tmp_path / "path-root"
    events, lineage, _, _ = _baseline(root)
    rejected = _rejected(
        _run(
            "init",
            "--state-events",
            tmp_path / "stale-store",
            "--lineage-events",
            events,
            "--artifact-root",
            root,
            "--sealed-study",
            root / "artifacts" / "study-1.json",
            "--expected-lineage-tip-sha256",
            "0" * 64,
        )
    )
    assert "stale or forked lineage tip" in rejected["error"]
    assert not (tmp_path / "stale-store").exists()

    store = tmp_path / "path-store"
    current = _accepted(
        _run(
            "init",
            "--state-events",
            store,
            "--lineage-events",
            events,
            "--artifact-root",
            root,
            "--sealed-study",
            root / "artifacts" / "study-1.json",
            "--expected-lineage-tip-sha256",
            lineage.latest_event_sha256,
        )
    )
    request = root / "artifacts" / "request.json"
    request.write_text("{}\n", encoding="utf-8")
    tip, parent = _cas(current)
    current = _accepted(
        _run(
            "reserve-model",
            "--state-events",
            store,
            "--expected-tip-event-sha256",
            tip,
            "--expected-parent-state-sha256",
            parent,
            "--lineage-events",
            events,
            "--artifact-root",
            root,
            "--invocation-id",
            "symlink-model",
            "--request-artifact",
            request,
            "--event-at-utc",
            "2026-07-20T00:01:00Z",
        )
    )
    response_input = root / "response.tmp"
    response_input.write_text("[]", encoding="utf-8")
    response_dir = root / "model-responses"
    response_dir.mkdir()
    outside = tmp_path / "outside-response"
    outside.write_text("do not overwrite", encoding="utf-8")
    response_link = response_dir / "response.json"
    response_link.symlink_to(outside)
    tip, parent = _cas(current)
    rejected = _rejected(
        _run(
            "record-response",
            "--state-events",
            store,
            "--expected-tip-event-sha256",
            tip,
            "--expected-parent-state-sha256",
            parent,
            "--artifact-root",
            root,
            "--invocation-id",
            "symlink-model",
            "--response-input",
            response_input,
            "--response-artifact",
            response_link,
            "--event-at-utc",
            "2026-07-20T00:02:00Z",
        )
    )
    assert "regular file" in rejected["error"]
    assert outside.read_text(encoding="utf-8") == "do not overwrite"
    assert verify_state_store(store)["event_count"] == 2
