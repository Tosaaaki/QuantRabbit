#!/usr/bin/env python3
"""Build one bounded, research-only DOJO AI trainer packet artifact.

The command is a file-to-file reducer.  It never calls a model, replay runner,
broker, execution gateway, or order API, and it cannot grant live permission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.dojo_ai_trainer_packet import (  # noqa: E402
    MAX_BOUND_ARTIFACT_BYTES,
    DojoAITrainerPacketError,
    build_trainer_packet,
    canonical_packet_bytes,
)
from quant_rabbit.dojo_ai_tuning_state import (  # noqa: E402
    DojoAITuningStateError,
    verify_state_store,
)
from quant_rabbit.dojo_candidate_lineage_registry import (  # noqa: E402
    CandidateLineageError,
    CandidateLineageSnapshot,
    verify_registry,
)
from quant_rabbit.dojo_terminal_handoff import (  # noqa: E402
    DojoTerminalHandoffError,
    canonical_sha256 as canonical_handoff_sha256,
    verify_receipt_store,
)


RECEIPT_CONTRACT = "QR_DOJO_AI_TRAINER_PACKET_CLI_RECEIPT_V1"
SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class DojoAITrainerPacketCliError(ValueError):
    """A CLI artifact, compare-and-check token, or output path is unsafe."""


@dataclass(frozen=True)
class _LoadedArtifact:
    value: Any
    sha256: str
    size_bytes: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-artifact", type=Path, required=True)
    parser.add_argument("--evaluation-artifact", type=Path, required=True)
    parser.add_argument("--cells-artifact", type=Path, required=True)
    parser.add_argument("--lineage-events", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--tuning-state-events", type=Path, required=True)
    parser.add_argument("--expected-lineage-tip-sha256", required=True)
    parser.add_argument("--expected-tuning-tip-event-sha256", required=True)
    parser.add_argument("--expected-tuning-state-sha256", required=True)
    parser.add_argument("--handoff-receipt-events", type=Path, required=True)
    parser.add_argument("--expected-handoff-receipt-tip-sha256", required=True)
    parser.add_argument("--drive-evidence-refs-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DojoAITrainerPacketCliError(f"{label} must be a lowercase SHA-256")
    return value


def _reject_constant(token: str) -> None:
    raise DojoAITrainerPacketCliError(
        f"input artifact contains forbidden non-finite JSON token {token}"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DojoAITrainerPacketCliError(
                "input artifact contains a duplicate JSON key"
            )
        result[key] = value
    return result


def _validate_json(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DojoAITrainerPacketCliError(
                "input artifact contains a non-finite number"
            )
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise DojoAITrainerPacketCliError(
                "input artifact contains a non-string object key"
            )
        for item in value.values():
            _validate_json(item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _validate_json(item)
        return
    raise DojoAITrainerPacketCliError("input artifact is not strict JSON")


def _read_json_artifact(path: Path, *, label: str) -> _LoadedArtifact:
    candidate = path.absolute()
    try:
        expected = candidate.lstat()
    except OSError as exc:
        raise DojoAITrainerPacketCliError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(expected.st_mode)
        or stat.S_ISLNK(expected.st_mode)
        or expected.st_size <= 0
    ):
        raise DojoAITrainerPacketCliError(f"{label} must be a nonempty regular file")
    if expected.st_size > MAX_BOUND_ARTIFACT_BYTES:
        raise DojoAITrainerPacketCliError(f"{label} exceeds the artifact size limit")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(candidate, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            before = os.fstat(handle.fileno())
            raw = handle.read(MAX_BOUND_ARTIFACT_BYTES + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise DojoAITrainerPacketCliError(f"{label} could not be read safely") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_dev != expected.st_dev
        or before.st_ino != expected.st_ino
        or len(raw) != before.st_size
    ):
        raise DojoAITrainerPacketCliError(f"{label} changed while being read")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except DojoAITrainerPacketCliError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DojoAITrainerPacketCliError(f"{label} is not strict JSON") from exc
    _validate_json(value)
    return _LoadedArtifact(
        value=value,
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
    )


def _require_terminal_handoff_binding(
    *,
    receipt: Mapping[str, Any],
    lineage: CandidateLineageSnapshot,
    run: _LoadedArtifact,
    evaluation: _LoadedArtifact,
    cells: _LoadedArtifact,
) -> None:
    """Require one exact full-bundle commit marker for the current lineage."""

    if not lineage.results or len(lineage.studies) != len(lineage.results):
        raise DojoAITrainerPacketCliError(
            "trainer packet requires a terminal lineage result"
        )
    if not lineage.events or lineage.events[-1]["event_type"] != "RESULT_BOUND":
        raise DojoAITrainerPacketCliError(
            "trainer packet requires the latest lineage RESULT_BOUND event"
        )
    latest_result = lineage.results[-1]
    latest_event = lineage.events[-1]
    if latest_event["body"] != latest_result:
        raise DojoAITrainerPacketCliError(
            "latest lineage RESULT_BOUND body is not exact"
        )

    terminal = receipt["terminal_bundle"]
    for name, loaded in (
        ("run", run),
        ("evaluation", evaluation),
        ("cells", cells),
    ):
        reference = terminal[name]
        if (
            reference["artifact_sha256"] != loaded.sha256
            or reference["artifact_size_bytes"] != loaded.size_bytes
        ):
            raise DojoAITrainerPacketCliError(
                f"latest hand-off receipt does not bind exact {name} artifact bytes"
            )
    if not isinstance(run.value, Mapping) or not isinstance(evaluation.value, Mapping):
        raise DojoAITrainerPacketCliError(
            "terminal run and evaluation artifacts must be JSON objects"
        )
    semantic_expected = {
        "run_sha256": run.value.get("run_sha256"),
        "evaluation_sha256": evaluation.value.get("evaluation_sha256"),
        "study_sha256": run.value.get("study_sha256"),
    }
    if any(terminal[key] != value for key, value in semantic_expected.items()):
        raise DojoAITrainerPacketCliError(
            "latest hand-off receipt semantic run/evaluation/study binding drifted"
        )
    if evaluation.value.get("study_sha256") != semantic_expected["study_sha256"]:
        raise DojoAITrainerPacketCliError(
            "terminal evaluation belongs to another hand-off study"
        )

    expected_after = {
        "registry_id": lineage.registry_id,
        "lineage_prefix": lineage.lineage_prefix,
        "latest_sequence": lineage.latest_sequence,
        "latest_event_sha256": lineage.latest_event_sha256,
        "attempt_ordinal": latest_result["attempt_ordinal"],
        "study_sha256": latest_result["study_sha256"],
        "evaluation_sha256": latest_result["evaluation_sha256"],
        "evaluation_artifact_sha256": latest_result["evaluation_artifact_sha256"],
        "evaluation_artifact_size_bytes": latest_result[
            "evaluation_artifact_size_bytes"
        ],
        "result_event_sha256": latest_event["event_sha256"],
        "result_event_sequence": latest_event["sequence"],
        "result_binding_sha256": canonical_handoff_sha256(latest_result),
    }
    if receipt["lineage_after"] != expected_after:
        raise DojoAITrainerPacketCliError(
            "latest hand-off receipt does not bind the exact current lineage result"
        )


def _open_output_parent(path: Path) -> tuple[int, Path]:
    target = path.absolute()
    if target.name in {"", ".", ".."}:
        raise DojoAITrainerPacketCliError("output path is invalid")
    try:
        resolved_parent = target.parent.resolve(strict=True)
        expected = target.parent.absolute().lstat()
    except OSError as exc:
        raise DojoAITrainerPacketCliError("output parent is unavailable") from exc
    if target.parent.absolute() != resolved_parent:
        raise DojoAITrainerPacketCliError("output parent must not traverse a symlink")
    if not stat.S_ISDIR(expected.st_mode) or stat.S_ISLNK(expected.st_mode):
        raise DojoAITrainerPacketCliError("output parent must be a real directory")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(resolved_parent, flags)
        actual = os.fstat(descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise DojoAITrainerPacketCliError(
            "output parent could not be opened safely"
        ) from exc
    assert descriptor is not None
    if (
        not stat.S_ISDIR(actual.st_mode)
        or actual.st_dev != expected.st_dev
        or actual.st_ino != expected.st_ino
    ):
        os.close(descriptor)
        raise DojoAITrainerPacketCliError("output parent changed while being opened")
    return descriptor, target


def _write_exclusive(path: Path, payload: bytes) -> None:
    parent_fd, target = _open_output_parent(path)
    descriptor: int | None = None
    created = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target.name, flags, 0o600, dir_fd=parent_fd)
        created = True
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            written = handle.write(payload)
            if written != len(payload):
                raise DojoAITrainerPacketCliError("packet output write was incomplete")
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(parent_fd)
    except FileExistsError as exc:
        raise DojoAITrainerPacketCliError(
            "packet output already exists; refusing to overwrite"
        ) from exc
    except BaseException:
        if created:
            try:
                os.unlink(target.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _canonical_line(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    expected_lineage_tip = _sha(
        args.expected_lineage_tip_sha256,
        label="expected lineage tip",
    )
    expected_store_tip = _sha(
        args.expected_tuning_tip_event_sha256,
        label="expected tuning store tip",
    )
    expected_state_sha = _sha(
        args.expected_tuning_state_sha256,
        label="expected tuning state",
    )
    expected_handoff_tip = _sha(
        args.expected_handoff_receipt_tip_sha256,
        label="expected hand-off receipt tip",
    )

    before_store = verify_state_store(args.tuning_state_events)
    if before_store["latest_event_sha256"] != expected_store_tip:
        raise DojoAITrainerPacketCliError("stale or forked tuning store event tip")
    if before_store["latest_state"]["state_sha256"] != expected_state_sha:
        raise DojoAITrainerPacketCliError("stale or forked tuning parent state")
    before_lineage = verify_registry(
        args.lineage_events, artifact_root=args.artifact_root
    )
    if before_lineage.latest_event_sha256 != expected_lineage_tip:
        raise DojoAITrainerPacketCliError("stale or forked lineage event tip")

    # The full-bundle hand-off is the downstream commit marker.  Verify its
    # complete append-only store before opening any packet input artifact; a
    # bare RESULT_BOUND event is deliberately insufficient.
    before_handoffs = verify_receipt_store(args.handoff_receipt_events)
    if not before_handoffs:
        raise DojoAITrainerPacketCliError(
            "terminal RESULT_BOUND has no full-bundle hand-off receipt"
        )
    latest_handoff = before_handoffs[-1]
    if latest_handoff["receipt_sha256"] != expected_handoff_tip:
        raise DojoAITrainerPacketCliError(
            "stale or forked terminal hand-off receipt tip"
        )

    run_artifact = _read_json_artifact(args.run_artifact, label="terminal run artifact")
    evaluation_artifact = _read_json_artifact(
        args.evaluation_artifact, label="terminal evaluation artifact"
    )
    cells_artifact = _read_json_artifact(
        args.cells_artifact, label="terminal cells artifact"
    )
    _require_terminal_handoff_binding(
        receipt=latest_handoff,
        lineage=before_lineage,
        run=run_artifact,
        evaluation=evaluation_artifact,
        cells=cells_artifact,
    )
    drive_refs_artifact = _read_json_artifact(
        args.drive_evidence_refs_artifact,
        label="Drive evidence references artifact",
    )
    drive_refs = drive_refs_artifact.value
    if not isinstance(drive_refs, list) or not drive_refs:
        raise DojoAITrainerPacketCliError(
            "at least one remotely verified Drive evidence reference is required"
        )
    packet = build_trainer_packet(
        run=run_artifact.value,
        evaluation=evaluation_artifact.value,
        cells=cells_artifact.value,
        lineage_events_dir=args.lineage_events,
        artifact_root=args.artifact_root,
        tuning_state=before_store["latest_state"],
        drive_evidence_refs=drive_refs,
    )
    if packet["source_bindings"]["lineage_tip_sha256"] != expected_lineage_tip:
        raise DojoAITrainerPacketCliError("packet lineage binding drifted")
    if packet["source_bindings"]["tuning_state_sha256"] != expected_state_sha:
        raise DojoAITrainerPacketCliError("packet tuning-state binding drifted")

    # Re-read both append-only sources before publishing, so a packet is never
    # presented as the current trainer input if either CAS surface advanced
    # while the bounded result artifacts were being reduced.
    after_store = verify_state_store(args.tuning_state_events)
    after_lineage = verify_registry(
        args.lineage_events, artifact_root=args.artifact_root
    )
    after_handoffs = verify_receipt_store(args.handoff_receipt_events)
    if (
        after_store["latest_event_sha256"] != expected_store_tip
        or after_store["latest_state"]["state_sha256"] != expected_state_sha
    ):
        raise DojoAITrainerPacketCliError(
            "tuning state changed while packet was being built"
        )
    if after_lineage.latest_event_sha256 != expected_lineage_tip:
        raise DojoAITrainerPacketCliError(
            "candidate lineage changed while packet was being built"
        )
    if (
        not after_handoffs
        or after_handoffs[-1]["receipt_sha256"] != expected_handoff_tip
        or after_handoffs != before_handoffs
    ):
        raise DojoAITrainerPacketCliError(
            "terminal hand-off receipt store changed while packet was being built"
        )

    payload = canonical_packet_bytes(packet) + b"\n"
    _write_exclusive(args.output, payload)
    return {
        "contract": RECEIPT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "status": "WRITTEN",
        "output_path": str(args.output.absolute()),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "output_size_bytes": len(payload),
        "packet_sha256": packet["packet_sha256"],
        "lineage_tip_sha256": expected_lineage_tip,
        "tuning_store_tip_event_sha256": expected_store_tip,
        "tuning_state_sha256": expected_state_sha,
        "terminal_handoff_receipt_sha256": expected_handoff_tip,
        "classification": packet["classification"],
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }


def main() -> int:
    args = _parser().parse_args()
    try:
        receipt = _run(args)
    except (
        DojoAITrainerPacketError,
        DojoAITrainerPacketCliError,
        DojoAITuningStateError,
        CandidateLineageError,
        DojoTerminalHandoffError,
    ) as exc:
        print(
            _canonical_line(
                {
                    "status": "REJECTED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "live_permission": False,
                    "order_authority": "NONE",
                    "broker_mutation_allowed": False,
                }
            ),
            file=sys.stderr,
        )
        return 2
    except OSError:
        print(
            _canonical_line(
                {
                    "status": "REJECTED",
                    "error_type": "LOCAL_IO_ERROR",
                    "error": "local artifact operation failed",
                    "live_permission": False,
                    "order_authority": "NONE",
                    "broker_mutation_allowed": False,
                }
            ),
            file=sys.stderr,
        )
        return 2
    print(_canonical_line(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
