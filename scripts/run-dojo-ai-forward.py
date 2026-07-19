#!/usr/bin/env python3
"""Operate the two-stage prospective DOJO AI prompt experiment."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_rabbit.broker.oanda import OandaReadOnlyClient  # noqa: E402
from quant_rabbit.dojo_ai_forward import (  # noqa: E402
    DojoAIForwardError,
    OFFICIAL_OANDA_BASE_URL,
    build_cell_response_failure,
    build_cell_response_seal,
    build_day_requests_from_capture,
    build_missing_day_seal,
    build_phase_index,
    build_precommit,
    build_start_receipt,
    build_source_capture_from_request,
    prepare_day_request,
    validate_cell_terminal,
    validate_day_seal,
    validate_phase_index,
    validate_precommit,
    validate_start_receipt,
    validate_source_capture,
    validate_source_request,
    validate_supersession,
    verify_source_bindings_against_repo,
)
from quant_rabbit.dojo_ai_truth import (  # noqa: E402
    build_day_score as build_truth_day_score,
    build_phase_score,
    build_truth_bundle,
    build_truth_capture,
    build_truth_request,
    issue_day_truth_capability,
    issue_phase_truth_capability,
    validate_day_score as validate_truth_day_score,
    validate_phase_score,
    validate_truth_bundle_with_capture,
    validate_truth_capture,
    validate_truth_request,
)
from quant_rabbit.dojo_ai_validity import (  # noqa: E402
    REGISTRY_DIRNAME,
    append_artifact_commit,
    append_invalidation,
    assert_artifacts_valid,
    initialize_registry,
    status_artifact as validity_status_artifact,
    verify_registry,
)
from quant_rabbit.dojo_prompt_phase import assert_locked_preregistration  # noqa: E402


DEFAULT_REGISTRY = REPO / "research/registries/dojo_prompt_experiment_v1.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc_text(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError("AI artifact timestamp is not exact UTC text")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise RuntimeError("AI artifact timestamp is invalid") from exc


def _load_parents(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if _superseded_status(run_dir) is not None:
        raise RuntimeError("AI forward run is terminally superseded")
    precommit = validate_precommit(_strict_json(run_dir / "precommit.json"))
    verify_source_bindings_against_repo(precommit["source_bindings"], REPO)
    start = validate_start_receipt(_strict_json(run_dir / "start.json"), precommit)
    assert_artifacts_valid(run_dir, ("precommit", "start"))
    return precommit, start


def _previous_day(run_dir: Path, ordinal: int) -> dict[str, Any] | None:
    if ordinal == 1:
        return None
    return _strict_json(run_dir / "days" / f"day-{ordinal - 1:03d}.json")


def _precommit(args: argparse.Namespace) -> dict[str, Any]:
    with _run_lock(args.run_dir):
        if _superseded_status(args.run_dir) is not None:
            raise RuntimeError("AI forward run is terminally superseded")
        registry = assert_locked_preregistration(_strict_json(args.registry))
        prompt_texts = {
            row["variant_id"]: (REPO / row["prompt_path"]).read_text(encoding="utf-8")
            for row in registry["variants"]
        }
        artifact = build_precommit(
            registry,
            prompt_texts,
            _strict_json(args.spec),
            now_utc=_now(),
        )
        verify_source_bindings_against_repo(artifact["source_bindings"], REPO)
        _write_json_new_or_same(
            args.run_dir / "precommit.json", artifact, root=args.run_dir
        )
        return artifact


def _start(args: argparse.Namespace) -> dict[str, Any]:
    with _run_lock(args.run_dir):
        if _superseded_status(args.run_dir) is not None:
            raise RuntimeError("AI forward run is terminally superseded")
        precommit = validate_precommit(_strict_json(args.run_dir / "precommit.json"))
        receipt = build_start_receipt(precommit, now_utc=_now())
        _write_json_new_or_same(args.run_dir / "start.json", receipt, root=args.run_dir)
        initialize_registry(
            args.run_dir,
            precommit_path=args.run_dir / "precommit.json",
            start_path=args.run_dir / "start.json",
            created_at_utc=_now(),
        )
        return receipt


def _day_parent_ids(ordinal: int) -> list[str]:
    parents = ["precommit", "start"]
    if ordinal > 1:
        parents.append(f"day/{ordinal - 1:03d}/seal")
    return parents


def _commit_source_request(run_dir: Path, ordinal: int) -> None:
    path = run_dir / "source-requests" / f"day-{ordinal:03d}.json"
    append_artifact_commit(
        run_dir,
        logical_id=f"day/{ordinal:03d}/source-request",
        artifact_path=path,
        parent_logical_ids=_day_parent_ids(ordinal),
        committed_at_utc=_now(),
    )


def _commit_source_capture(run_dir: Path, ordinal: int) -> None:
    append_artifact_commit(
        run_dir,
        logical_id=f"day/{ordinal:03d}/source-capture",
        artifact_path=run_dir / "source-captures" / f"day-{ordinal:03d}.json",
        parent_logical_ids=(f"day/{ordinal:03d}/source-request",),
        committed_at_utc=_now(),
    )


def _commit_day(run_dir: Path, ordinal: int, day: Mapping[str, Any]) -> None:
    parents = _day_parent_ids(ordinal)
    if day.get("source_capture_sha256") is not None:
        parents.append(f"day/{ordinal:03d}/source-capture")
    else:
        request_path = run_dir / "source-requests" / f"day-{ordinal:03d}.json"
        if request_path.is_file() and not request_path.is_symlink():
            parents.append(f"day/{ordinal:03d}/source-request")
    append_artifact_commit(
        run_dir,
        logical_id=f"day/{ordinal:03d}/seal",
        artifact_path=run_dir / "days" / f"day-{ordinal:03d}.json",
        parent_logical_ids=parents,
        committed_at_utc=_now(),
    )


def _commit_terminal(run_dir: Path, ordinal: int, cell_id: str) -> None:
    append_artifact_commit(
        run_dir,
        logical_id=f"day/{ordinal:03d}/cell/{cell_id}/terminal",
        artifact_path=_terminal_path(run_dir, ordinal, cell_id),
        parent_logical_ids=(f"day/{ordinal:03d}/seal",),
        committed_at_utc=_now(),
    )


def _terminal_logical_ids(day: Mapping[str, Any]) -> list[str]:
    if day["state"] != "REQUESTS_SEALED":
        return []
    return [
        f"day/{day['ordinal']:03d}/cell/{cell['cell_id']}/terminal"
        for cell in day["schedule"]["cells"]
    ]


def _commit_phase_index(run_dir: Path, days: list[dict[str, Any]]) -> None:
    parents = ["precommit", "start"]
    parents.extend(f"day/{day['ordinal']:03d}/seal" for day in days)
    for day in days:
        parents.extend(_terminal_logical_ids(day))
    append_artifact_commit(
        run_dir,
        logical_id="phase/index",
        artifact_path=run_dir / "phase-index.json",
        parent_logical_ids=parents,
        committed_at_utc=_now(),
    )


def _truth_logical(ordinal: int, stage: str) -> str:
    return f"day/{ordinal:03d}/truth/{stage}"


def _issue_truth_capability(
    run_dir: Path,
    registry: Mapping[str, Any],
    precommit: Mapping[str, Any],
    start: Mapping[str, Any],
    days: list[dict[str, Any]],
    day: Mapping[str, Any],
    terminals: list[dict[str, Any]],
    *,
    required_stages: tuple[str, ...] = (),
):
    ordinal = int(day["ordinal"])
    previous = days[ordinal - 2] if ordinal > 1 else None
    required_ids = tuple(_truth_logical(ordinal, stage) for stage in required_stages)
    return issue_day_truth_capability(
        run_dir,
        registry,
        precommit,
        start,
        previous,
        day,
        terminals,
        required_committed_logical_ids=required_ids,
    )


def _commit_truth_stage(
    run_dir: Path,
    day: Mapping[str, Any],
    stage: str,
) -> None:
    ordinal = int(day["ordinal"])
    terminal_ids = _terminal_logical_ids(day)
    if stage == "request":
        parents = ["precommit", f"day/{ordinal:03d}/seal", *terminal_ids]
    elif stage == "capture":
        parents = [_truth_logical(ordinal, "request")]
    elif stage == "bundle":
        parents = [
            "precommit",
            f"day/{ordinal:03d}/seal",
            *terminal_ids,
            _truth_logical(ordinal, "request"),
            _truth_logical(ordinal, "capture"),
        ]
    elif stage == "score":
        parents = [
            "precommit",
            f"day/{ordinal:03d}/seal",
            *terminal_ids,
            _truth_logical(ordinal, "bundle"),
        ]
    else:
        raise RuntimeError("unknown AI truth validity stage")
    append_artifact_commit(
        run_dir,
        logical_id=_truth_logical(ordinal, stage),
        artifact_path=run_dir / "truth" / f"day-{ordinal:03d}" / f"{stage}.json",
        parent_logical_ids=parents,
        committed_at_utc=_now(),
    )


def _commit_phase_score(run_dir: Path, days: list[dict[str, Any]]) -> None:
    score_parents = [
        _truth_logical(int(day["ordinal"]), "score")
        for day in days
        if day["state"] == "REQUESTS_SEALED"
    ]
    append_artifact_commit(
        run_dir,
        logical_id="phase/score",
        artifact_path=run_dir / "phase-score.json",
        parent_logical_ids=("precommit", "phase/index", *score_parents),
        committed_at_utc=_now(),
    )


def _collect_day(args: argparse.Namespace) -> dict[str, Any]:
    with _run_lock(args.run_dir):
        registry = assert_locked_preregistration(_strict_json(args.registry))
        precommit, start = _load_parents(args.run_dir)
        days = _validated_days(args.run_dir, registry, precommit, start)
        if args.ordinal < 1 or args.ordinal > len(days) + 1:
            raise RuntimeError("AI source collection ordinal is not the next chain day")
        previous = days[args.ordinal - 2] if args.ordinal > 1 else None
        day_path = args.run_dir / "days" / f"day-{args.ordinal:03d}.json"
        capture_path = args.run_dir / "source-captures" / f"day-{args.ordinal:03d}.json"
        request_path = args.run_dir / "source-requests" / f"day-{args.ordinal:03d}.json"
        if day_path.is_file():
            day = validate_day_seal(
                _strict_json(day_path),
                registry,
                precommit,
                start,
                previous,
                expected_ordinal=args.ordinal,
            )
            capture = validate_source_capture(
                _strict_json(capture_path), precommit, start, previous
            )
            if day.get("source_capture_sha256") != capture["source_capture_sha256"]:
                raise RuntimeError("AI day does not bind its persisted source capture")
            _commit_source_request(args.run_dir, args.ordinal)
            _commit_source_capture(args.run_dir, args.ordinal)
            _commit_day(args.run_dir, args.ordinal, day)
            return day
        if capture_path.is_file():
            _commit_source_request(args.run_dir, args.ordinal)
            capture = validate_source_capture(
                _strict_json(capture_path), precommit, start, previous
            )
            _commit_source_capture(args.run_dir, args.ordinal)
            seal = build_day_requests_from_capture(
                registry,
                precommit,
                start,
                previous,
                capture,
            )
            _write_json_new_or_same(day_path, seal, root=args.run_dir)
            _commit_day(args.run_dir, args.ordinal, seal)
            return seal
        if request_path.is_file():
            request = validate_source_request(
                _strict_json(request_path), precommit, start, previous
            )
            if _now() > _parse_utc_text(request["source_seal_deadline_utc"]):
                raise RuntimeError("AI source request expired before transport")
        else:
            request = prepare_day_request(
                precommit,
                start,
                previous,
                ordinal=args.ordinal,
                now_utc=_now(),
            )
            _write_json_new_or_same(request_path, request, root=args.run_dir)
        _commit_source_request(args.run_dir, args.ordinal)
        client = OandaReadOnlyClient()
        if client.base_url != OFFICIAL_OANDA_BASE_URL:
            raise RuntimeError(
                "AI source requires the official OANDA production HTTPS host"
            )
        response = _get_with_retry(
            client,
            request["path"],
            request["query"],
            attempts=args.attempts,
        )
        capture = build_source_capture_from_request(
            precommit,
            start,
            previous,
            request,
            response,
            acquired_at_utc=_now(),
        )
        _write_json_new_or_same(capture_path, capture, root=args.run_dir)
        _commit_source_capture(args.run_dir, args.ordinal)
        seal = build_day_requests_from_capture(
            registry,
            precommit,
            start,
            previous,
            capture,
        )
        _write_json_new_or_same(day_path, seal, root=args.run_dir)
        _commit_day(args.run_dir, args.ordinal, seal)
        return seal


def _seal_missing(args: argparse.Namespace) -> dict[str, Any]:
    with _run_lock(args.run_dir):
        registry = assert_locked_preregistration(_strict_json(args.registry))
        precommit, start = _load_parents(args.run_dir)
        days = _validated_days(args.run_dir, registry, precommit, start)
        if args.ordinal <= len(days):
            return days[args.ordinal - 1]
        if args.ordinal != len(days) + 1:
            raise RuntimeError("AI missing source ordinal is not the next chain day")
        previous = days[-1] if days else None
        capture_path = args.run_dir / "source-captures" / f"day-{args.ordinal:03d}.json"
        failed_capture_sha: str | None = None
        if capture_path.is_file():
            request_path = (
                args.run_dir / "source-requests" / f"day-{args.ordinal:03d}.json"
            )
            if request_path.is_file() and not request_path.is_symlink():
                _commit_source_request(args.run_dir, args.ordinal)
            capture = validate_source_capture(
                _strict_json(capture_path), precommit, start, previous
            )
            _commit_source_capture(args.run_dir, args.ordinal)
            try:
                seal = build_day_requests_from_capture(
                    registry,
                    precommit,
                    start,
                    previous,
                    capture,
                )
            except DojoAIForwardError:
                failed_capture_sha = capture["source_capture_sha256"]
            else:
                _write_json_new_or_same(
                    args.run_dir / "days" / f"day-{args.ordinal:03d}.json",
                    seal,
                    root=args.run_dir,
                )
                _commit_day(args.run_dir, args.ordinal, seal)
                return seal
        receipt = build_missing_day_seal(
            precommit,
            start,
            previous,
            ordinal=args.ordinal,
            now_utc=_now(),
            failed_source_capture_sha256=failed_capture_sha,
        )
        _write_json_new_or_same(
            args.run_dir / "days" / f"day-{args.ordinal:03d}.json",
            receipt,
            root=args.run_dir,
        )
        request_path = args.run_dir / "source-requests" / f"day-{args.ordinal:03d}.json"
        if request_path.is_file() and not request_path.is_symlink():
            _commit_source_request(args.run_dir, args.ordinal)
        _commit_day(args.run_dir, args.ordinal, receipt)
        return receipt


def _load_day_context(
    run_dir: Path,
    registry: Mapping[str, Any],
    precommit: Mapping[str, Any],
    start: Mapping[str, Any],
    ordinal: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    days = _validated_days(run_dir, registry, precommit, start)
    if ordinal < 1 or ordinal > len(days):
        raise RuntimeError("AI response day is not sealed")
    previous = days[ordinal - 2] if ordinal > 1 else None
    day = days[ordinal - 1]
    return previous, day


def _terminal_path(run_dir: Path, ordinal: int, cell_id: str) -> Path:
    return run_dir / "responses" / f"day-{ordinal:03d}" / f"{cell_id}.json"


def _seal_cell_response(args: argparse.Namespace) -> dict[str, Any]:
    with _run_lock(args.run_dir):
        registry = assert_locked_preregistration(_strict_json(args.registry))
        precommit, start = _load_parents(args.run_dir)
        previous, day = _load_day_context(
            args.run_dir, registry, precommit, start, args.ordinal
        )
        response = _strict_json(args.response_json)
        path = _terminal_path(args.run_dir, args.ordinal, args.cell_id)
        if path.is_file():
            terminal = validate_cell_terminal(
                _strict_json(path),
                registry,
                precommit,
                start,
                previous,
                day,
            )
            if (
                terminal["state"] != "DIAGNOSTIC_IMPORTED_RESPONSE"
                or terminal["response_receipt"]["response"] != response
            ):
                raise RuntimeError("immutable cell terminal already differs")
            _commit_terminal(args.run_dir, args.ordinal, args.cell_id)
            return terminal
        terminal = build_cell_response_seal(
            registry,
            precommit,
            start,
            previous,
            day,
            response,
            cell_id=args.cell_id,
            now_utc=_now(),
        )
        _write_json_new_or_same(path, terminal, root=args.run_dir)
        _commit_terminal(args.run_dir, args.ordinal, args.cell_id)
        return terminal


def _seal_missing_responses(args: argparse.Namespace) -> dict[str, Any]:
    with _run_lock(args.run_dir):
        registry = assert_locked_preregistration(_strict_json(args.registry))
        precommit, start = _load_parents(args.run_dir)
        previous, day = _load_day_context(
            args.run_dir, registry, precommit, start, args.ordinal
        )
        if day["state"] == "MISSING_SOURCE_DEADLINE":
            return {
                "contract": "QR_DOJO_AI_FORWARD_MISSING_RESPONSE_BATCH_V3",
                "ordinal": args.ordinal,
                "created_count": 0,
                "existing_count": 0,
                "source_missing_cell_count": 3,
                "live_permission": False,
            }
        now = _now()
        created = 0
        existing = 0
        for scheduled in day["schedule"]["cells"]:
            cell_id = scheduled["cell_id"]
            path = _terminal_path(args.run_dir, args.ordinal, cell_id)
            if path.is_file():
                validate_cell_terminal(
                    _strict_json(path),
                    registry,
                    precommit,
                    start,
                    previous,
                    day,
                )
                _commit_terminal(args.run_dir, args.ordinal, cell_id)
                existing += 1
                continue
            terminal = build_cell_response_failure(
                precommit,
                day,
                cell_id=cell_id,
                now_utc=now,
            )
            _write_json_new_or_same(path, terminal, root=args.run_dir)
            _commit_terminal(args.run_dir, args.ordinal, cell_id)
            created += 1
        return {
            "contract": "QR_DOJO_AI_FORWARD_MISSING_RESPONSE_BATCH_V3",
            "ordinal": args.ordinal,
            "created_count": created,
            "existing_count": existing,
            "source_missing_cell_count": 0,
            "live_permission": False,
        }


def _validated_days(
    run_dir: Path,
    registry: Mapping[str, Any],
    precommit: Mapping[str, Any],
    start: Mapping[str, Any],
) -> list[dict[str, Any]]:
    paths = sorted((run_dir / "days").glob("day-*.json"))
    previous = None
    valid: list[dict[str, Any]] = []
    for ordinal, path in enumerate(paths, start=1):
        if path.name != f"day-{ordinal:03d}.json" or path.is_symlink():
            raise RuntimeError("AI day path chain is not exact or is unsafe")
        seal = validate_day_seal(
            _strict_json(path),
            registry,
            precommit,
            start,
            previous,
            expected_ordinal=ordinal,
        )
        capture_path = run_dir / "source-captures" / f"day-{ordinal:03d}.json"
        capture_sha = seal.get("source_capture_sha256")
        if capture_sha is not None:
            if not capture_path.is_file() or capture_path.is_symlink():
                raise RuntimeError("AI day source capture is absent or unsafe")
            capture = validate_source_capture(
                _strict_json(capture_path), precommit, start, previous
            )
            if capture["source_capture_sha256"] != capture_sha:
                raise RuntimeError("AI day source capture binding drifted")
            request_path = run_dir / "source-requests" / f"day-{ordinal:03d}.json"
            request = validate_source_request(
                _strict_json(request_path), precommit, start, previous
            )
            if request != capture["request"]:
                raise RuntimeError("AI persisted source request differs from capture")
            assert_artifacts_valid(
                run_dir,
                (
                    f"day/{ordinal:03d}/source-request",
                    f"day/{ordinal:03d}/source-capture",
                ),
            )
        elif seal["state"] == "REQUESTS_SEALED" or capture_path.exists():
            raise RuntimeError("AI day and persisted capture are not joined")
        valid.append(seal)
        assert_artifacts_valid(run_dir, (f"day/{ordinal:03d}/seal",))
        previous = seal
    return valid


def _validated_terminals(
    run_dir: Path,
    registry: Mapping[str, Any],
    precommit: Mapping[str, Any],
    start: Mapping[str, Any],
    days: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_paths: set[Path] = set()
    terminals: list[dict[str, Any]] = []
    for ordinal, day in enumerate(days, start=1):
        if day["state"] != "REQUESTS_SEALED":
            continue
        previous = days[ordinal - 2] if ordinal > 1 else None
        for scheduled in day["schedule"]["cells"]:
            path = _terminal_path(run_dir, ordinal, scheduled["cell_id"])
            expected_paths.add(path)
            if not path.exists() and not path.is_symlink():
                continue
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("AI cell terminal path is unsafe")
            terminal = validate_cell_terminal(
                _strict_json(path),
                registry,
                precommit,
                start,
                previous,
                day,
            )
            if terminal["cell_id"] != scheduled["cell_id"]:
                raise RuntimeError("AI cell terminal filename identity drifted")
            assert_artifacts_valid(
                run_dir,
                (f"day/{ordinal:03d}/cell/{scheduled['cell_id']}/terminal",),
            )
            terminals.append(terminal)
    response_root = run_dir / "responses"
    if response_root.is_symlink():
        raise RuntimeError("AI response directory cannot be a symlink")
    actual_paths = (
        set(response_root.rglob("*.json")) if response_root.exists() else set()
    )
    unexpected = actual_paths - expected_paths
    if unexpected:
        raise RuntimeError("unexpected AI cell terminal artifact is present")
    return terminals


def _seal_phase_index(args: argparse.Namespace) -> dict[str, Any]:
    with _run_lock(args.run_dir):
        registry = assert_locked_preregistration(_strict_json(args.registry))
        precommit, start = _load_parents(args.run_dir)
        days = _validated_days(args.run_dir, registry, precommit, start)
        terminals = _validated_terminals(args.run_dir, registry, precommit, start, days)
        path = args.run_dir / "phase-index.json"
        if path.is_file():
            existing = validate_phase_index(
                _strict_json(path), registry, precommit, start, days, terminals
            )
            _commit_phase_index(args.run_dir, days)
            return existing
        index = build_phase_index(
            registry,
            precommit,
            start,
            days,
            terminals,
            now_utc=_now(),
        )
        _write_json_new_or_same(path, index, root=args.run_dir)
        _commit_phase_index(args.run_dir, days)
        return index


def _terminals_for_day(
    terminals: list[dict[str, Any]], day: Mapping[str, Any]
) -> list[dict[str, Any]]:
    expected = {cell["cell_id"] for cell in day["schedule"]["cells"]}
    selected = [row for row in terminals if row["cell_id"] in expected]
    if (
        day["state"] == "REQUESTS_SEALED"
        and {row["cell_id"] for row in selected} != expected
    ):
        raise RuntimeError("AI truth requires all three immutable cell terminals")
    return sorted(selected, key=lambda row: row["variant_id"])


def _collect_truth(args: argparse.Namespace) -> dict[str, Any]:
    with _run_lock(args.run_dir):
        registry = assert_locked_preregistration(_strict_json(args.registry))
        precommit, start = _load_parents(args.run_dir)
        days = _validated_days(args.run_dir, registry, precommit, start)
        if args.ordinal < 1 or args.ordinal > len(days):
            raise RuntimeError("AI truth ordinal is not a sealed day")
        day = days[args.ordinal - 1]
        if day["state"] != "REQUESTS_SEALED":
            raise RuntimeError("missing-source day has no market truth score")
        all_terminals = _validated_terminals(
            args.run_dir, registry, precommit, start, days
        )
        terminals = _terminals_for_day(all_terminals, day)
        truth_dir = args.run_dir / "truth" / f"day-{args.ordinal:03d}"
        request_path = truth_dir / "request.json"
        capture_path = truth_dir / "capture.json"
        bundle_path = truth_dir / "bundle.json"
        score_path = truth_dir / "score.json"
        base_capability = _issue_truth_capability(
            args.run_dir,
            registry,
            precommit,
            start,
            days,
            day,
            terminals,
        )
        if request_path.is_file():
            request = validate_truth_request(
                _strict_json(request_path),
                precommit,
                day,
                terminals,
                graph_capability=base_capability,
            )
        else:
            request = build_truth_request(
                precommit,
                day,
                terminals,
                now_utc=_now(),
                graph_capability=base_capability,
            )
            _write_json_new_or_same(request_path, request, root=args.run_dir)
        _commit_truth_stage(args.run_dir, day, "request")
        if capture_path.is_file():
            capture = validate_truth_capture(_strict_json(capture_path), request)
        else:
            if _now() > _parse_utc_text(request["truth_seal_deadline_utc"]):
                raise RuntimeError("AI truth request expired before transport")
            client = OandaReadOnlyClient()
            if client.base_url != OFFICIAL_OANDA_BASE_URL:
                raise RuntimeError(
                    "AI truth requires the official OANDA production HTTPS host"
                )
            response = _get_with_retry(
                client,
                request["path"],
                request["query"],
                attempts=args.attempts,
            )
            capture = build_truth_capture(request, response, acquired_at_utc=_now())
            _write_json_new_or_same(capture_path, capture, root=args.run_dir)
        _commit_truth_stage(args.run_dir, day, "capture")
        bundle_capability = _issue_truth_capability(
            args.run_dir,
            registry,
            precommit,
            start,
            days,
            day,
            terminals,
            required_stages=("request", "capture"),
        )
        if bundle_path.is_file():
            bundle = validate_truth_bundle_with_capture(
                _strict_json(bundle_path),
                precommit,
                day,
                terminals,
                capture,
                graph_capability=bundle_capability,
            )
        else:
            bundle = build_truth_bundle(
                precommit,
                day,
                terminals,
                capture,
                sealed_at_utc=_now(),
                graph_capability=bundle_capability,
            )
            _write_json_new_or_same(bundle_path, bundle, root=args.run_dir)
        _commit_truth_stage(args.run_dir, day, "bundle")
        score_capability = _issue_truth_capability(
            args.run_dir,
            registry,
            precommit,
            start,
            days,
            day,
            terminals,
            required_stages=("request", "capture", "bundle"),
        )
        score = build_truth_day_score(
            precommit,
            day,
            terminals,
            capture,
            bundle,
            scored_at_utc=_now(),
            graph_capability=score_capability,
        )
        _write_json_new_or_same(score_path, score, root=args.run_dir)
        _commit_truth_stage(args.run_dir, day, "score")
        return score


def _seal_phase_score(args: argparse.Namespace) -> dict[str, Any]:
    with _run_lock(args.run_dir):
        registry = assert_locked_preregistration(_strict_json(args.registry))
        precommit, start = _load_parents(args.run_dir)
        days = _validated_days(args.run_dir, registry, precommit, start)
        terminals = _validated_terminals(args.run_dir, registry, precommit, start, days)
        index = validate_phase_index(
            _strict_json(args.run_dir / "phase-index.json"),
            registry,
            precommit,
            start,
            days,
            terminals,
        )
        assert_artifacts_valid(args.run_dir, ("phase/index",))
        day_scores = _validated_truth_scores(
            args.run_dir,
            registry,
            precommit,
            start,
            days,
            terminals,
            require_all=True,
        )
        phase_capability = issue_phase_truth_capability(
            args.run_dir, precommit, index, day_scores
        )
        score_path = args.run_dir / "phase-score.json"
        if score_path.is_file() or score_path.is_symlink():
            existing = validate_phase_score(
                _strict_json(score_path),
                precommit,
                index,
                days,
                day_scores,
                graph_capability=phase_capability,
            )
            _commit_phase_score(args.run_dir, days)
            return existing
        phase_score = build_phase_score(
            precommit,
            index,
            days,
            day_scores,
            sealed_at_utc=_now(),
            graph_capability=phase_capability,
        )
        _write_json_new_or_same(
            args.run_dir / "phase-score.json", phase_score, root=args.run_dir
        )
        _commit_phase_score(args.run_dir, days)
        return phase_score


def _validated_truth_scores(
    run_dir: Path,
    registry: Mapping[str, Any],
    precommit: Mapping[str, Any],
    start: Mapping[str, Any],
    days: list[dict[str, Any]],
    terminals: list[dict[str, Any]],
    *,
    require_all: bool,
) -> list[dict[str, Any]]:
    expected = {
        day["ordinal"]: day for day in days if day["state"] == "REQUESTS_SEALED"
    }
    truth_root = run_dir / "truth"
    if truth_root.is_symlink():
        raise RuntimeError("AI truth directory cannot be a symlink")
    if truth_root.exists() and not truth_root.is_dir():
        raise RuntimeError("AI truth path is not a directory")
    actual_dirs: dict[int, Path] = {}
    if truth_root.is_dir():
        for candidate in truth_root.iterdir():
            if candidate.is_symlink() or not candidate.is_dir():
                raise RuntimeError("AI truth directory contains an unsafe entry")
            name = candidate.name
            if not name.startswith("day-") or len(name) != 7 or not name[4:].isdigit():
                raise RuntimeError("AI truth directory contains an unknown day")
            ordinal = int(name[4:])
            if ordinal not in expected or ordinal in actual_dirs:
                raise RuntimeError(
                    "AI truth directory is not bound to a sealed source day"
                )
            actual_dirs[ordinal] = candidate

    scores: list[dict[str, Any]] = []
    allowed_names = {"request.json", "capture.json", "bundle.json", "score.json"}
    for ordinal, truth_dir in sorted(actual_dirs.items()):
        entries = {candidate.name: candidate for candidate in truth_dir.iterdir()}
        if set(entries) - allowed_names:
            raise RuntimeError("AI truth day contains an unknown artifact")
        present = set(entries)
        stages = ["request.json", "capture.json", "bundle.json", "score.json"]
        for position, name in enumerate(stages):
            if name in present and not set(stages[:position]).issubset(present):
                raise RuntimeError(
                    "AI truth artifacts are not a contiguous evidence chain"
                )
        if "request.json" not in present:
            if present:
                raise RuntimeError("AI truth request is absent")
            continue
        day = expected[ordinal]
        selected = _terminals_for_day(terminals, day)
        base_capability = _issue_truth_capability(
            run_dir, registry, precommit, start, days, day, selected
        )
        request = validate_truth_request(
            _strict_json(entries["request.json"]),
            precommit,
            day,
            selected,
            graph_capability=base_capability,
        )
        assert_artifacts_valid(run_dir, (_truth_logical(ordinal, "request"),))
        if "capture.json" not in present:
            continue
        capture = validate_truth_capture(_strict_json(entries["capture.json"]), request)
        assert_artifacts_valid(run_dir, (_truth_logical(ordinal, "capture"),))
        if "bundle.json" not in present:
            continue
        bundle_capability = _issue_truth_capability(
            run_dir,
            registry,
            precommit,
            start,
            days,
            day,
            selected,
            required_stages=("request", "capture"),
        )
        bundle = validate_truth_bundle_with_capture(
            _strict_json(entries["bundle.json"]),
            precommit,
            day,
            selected,
            capture,
            graph_capability=bundle_capability,
        )
        assert_artifacts_valid(run_dir, (_truth_logical(ordinal, "bundle"),))
        if "score.json" not in present:
            continue
        score_capability = _issue_truth_capability(
            run_dir,
            registry,
            precommit,
            start,
            days,
            day,
            selected,
            required_stages=("request", "capture", "bundle"),
        )
        scores.append(
            validate_truth_day_score(
                _strict_json(entries["score.json"]),
                precommit,
                day,
                selected,
                capture,
                bundle,
                graph_capability=score_capability,
            )
        )
        assert_artifacts_valid(run_dir, (_truth_logical(ordinal, "score"),))
    if require_all and set(actual_dirs) != set(expected):
        raise RuntimeError("AI phase score requires every eligible truth day")
    if require_all and len(scores) != len(expected):
        raise RuntimeError("AI phase score requires every eligible day score")
    return scores


def _superseded_status(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "supersession.json"
    if not path.is_file() and not path.is_symlink():
        return None
    _assert_superseded_inventory(run_dir)
    precommit = validate_precommit(_strict_json(run_dir / "precommit.json"))
    start = validate_start_receipt(_strict_json(run_dir / "start.json"), precommit)
    supersession = validate_supersession(_strict_json(path), precommit, start)
    snapshot = verify_registry(run_dir)
    if (
        snapshot.event_count != 3
        or set(snapshot.committed) != {"precommit", "start", "run/supersession"}
        or snapshot.invalidated != {"start", "run/supersession"}
    ):
        raise RuntimeError("AI supersession validity graph drifted")
    committed = snapshot.committed["run/supersession"]
    if (
        committed["relative_path"] != "supersession.json"
        or committed["contract"] != supersession["contract"]
        or committed["artifact_sha256"] != supersession["supersession_sha256"]
        or committed["own_seal_field"] != "supersession_sha256"
        or committed["parent_logical_ids"] != ["precommit", "start"]
    ):
        raise RuntimeError("AI supersession artifact binding drifted")
    terminal_event = _strict_json(run_dir / REGISTRY_DIRNAME / "events" / "000002.json")
    if (
        terminal_event.get("event_sha256") != snapshot.latest_event_sha256
        or terminal_event.get("event_kind") != "ARTIFACT_INVALIDATED"
        or terminal_event.get("target_logical_id") != "start"
        or terminal_event.get("reason_code") != supersession["reason_code"]
        or terminal_event.get("evidence_sha256") != supersession["supersession_sha256"]
    ):
        raise RuntimeError("AI supersession invalidation binding drifted")
    validity = validity_status_artifact(run_dir)
    _assert_superseded_inventory(run_dir)
    return {
        "contract": "QR_DOJO_AI_FORWARD_STATUS_V3",
        "experiment_id": precommit["experiment_id"],
        "state": "SUPERSEDED_BEFORE_SOURCE",
        "reason_code": supersession["reason_code"],
        "superseded_at_utc": supersession["superseded_at_utc"],
        "supersession_sha256": supersession["supersession_sha256"],
        "successor_policy": supersession["successor_policy"],
        "sealed_day_count": 0,
        "request_day_count": 0,
        "missing_day_count": 0,
        "allocated_cell_count": supersession["allocated_cell_count"],
        "response_sealed_count": 0,
        "execution_failure_cell_count": 0,
        "diagnostic_import_cell_count": 0,
        "missing_response_cell_count": 0,
        "missing_source_cell_count": 0,
        "fixed_cell_count": 0,
        "next_ordinal": None,
        "promotion_eligible": False,
        "live_permission": False,
        "evidence_tier": precommit["evidence_tier"],
        "truth_day_score_count": 0,
        "phase_score_present": False,
        "validity_registry": validity,
    }


def _assert_superseded_inventory(run_dir: Path) -> None:
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise RuntimeError("AI supersession run directory is unsafe")
    expected_root = {
        ".ai-forward.lock",
        "precommit.json",
        "start.json",
        "supersession.json",
        REGISTRY_DIRNAME,
    }
    root_entries = {entry.name: entry for entry in run_dir.iterdir()}
    if (
        not {
            "precommit.json",
            "start.json",
            "supersession.json",
            REGISTRY_DIRNAME,
        }.issubset(root_entries)
        or set(root_entries) - expected_root
    ):
        raise RuntimeError("AI supersession run contains an unexpected artifact")
    for name in ("precommit.json", "start.json", "supersession.json"):
        entry = root_entries[name]
        if entry.is_symlink() or not entry.is_file():
            raise RuntimeError("AI supersession root artifact is unsafe")
    lock = root_entries.get(".ai-forward.lock")
    if lock is not None and (
        lock.is_symlink() or not lock.is_file() or lock.stat().st_size != 0
    ):
        raise RuntimeError("AI supersession lock artifact is unsafe")
    validity = root_entries[REGISTRY_DIRNAME]
    if validity.is_symlink() or not validity.is_dir():
        raise RuntimeError("AI supersession validity directory is unsafe")
    validity_entries = {entry.name: entry for entry in validity.iterdir()}
    if set(validity_entries) != {"events"}:
        raise RuntimeError("AI supersession validity inventory drifted")
    events = validity_entries["events"]
    if events.is_symlink() or not events.is_dir():
        raise RuntimeError("AI supersession event directory is unsafe")
    event_entries = {entry.name: entry for entry in events.iterdir()}
    if set(event_entries) != {"000000.json", "000001.json", "000002.json"}:
        raise RuntimeError("AI supersession event inventory drifted")
    if any(
        entry.is_symlink() or not entry.is_file() for entry in event_entries.values()
    ):
        raise RuntimeError("AI supersession event artifact is unsafe")


def _status(args: argparse.Namespace) -> dict[str, Any]:
    registry = assert_locked_preregistration(_strict_json(args.registry))
    superseded = _superseded_status(args.run_dir)
    if superseded is not None:
        return superseded
    precommit, start = _load_parents(args.run_dir)
    valid = _validated_days(args.run_dir, registry, precommit, start)
    terminals = _validated_terminals(args.run_dir, registry, precommit, start, valid)
    day_scores = _validated_truth_scores(
        args.run_dir,
        registry,
        precommit,
        start,
        valid,
        terminals,
        require_all=False,
    )
    phase_score_path = args.run_dir / "phase-score.json"
    phase_score = None
    if phase_score_path.is_file() or phase_score_path.is_symlink():
        index = validate_phase_index(
            _strict_json(args.run_dir / "phase-index.json"),
            registry,
            precommit,
            start,
            valid,
            terminals,
        )
        phase_score = validate_phase_score(
            _strict_json(phase_score_path),
            precommit,
            index,
            valid,
            day_scores,
            graph_capability=issue_phase_truth_capability(
                args.run_dir, precommit, index, day_scores
            ),
        )
        assert_artifacts_valid(args.run_dir, ("phase/index", "phase/score"))
    next_ordinal = len(valid) + 1
    response_count = sum(
        row["state"] == "EXECUTED_RESPONSE_SEALED" for row in terminals
    )
    execution_failure_count = sum(
        row["state"] == "MODEL_EXECUTION_FAILED" for row in terminals
    )
    diagnostic_import_count = sum(
        row["state"] == "DIAGNOSTIC_IMPORTED_RESPONSE" for row in terminals
    )
    missing_response_count = sum(
        row["state"] == "MISSING_RESPONSE_DEADLINE" for row in terminals
    )
    missing_source_cells = 3 * sum(
        row["state"] == "MISSING_SOURCE_DEADLINE" for row in valid
    )
    fixed_cells = (
        response_count
        + execution_failure_count
        + missing_response_count
        + missing_source_cells
    )
    if phase_score is not None:
        state = "PHASE_SCORED_DIAGNOSTIC"
    elif next_ordinal <= 30:
        state = "COLLECTING_SOURCE"
    elif fixed_cells < 90:
        state = "COLLECTING_RESPONSES"
    else:
        state = "RESPONSES_FIXED_AWAITING_MARKET_TRUTH"
    result: dict[str, Any] = {
        "contract": "QR_DOJO_AI_FORWARD_STATUS_V3",
        "experiment_id": precommit["experiment_id"],
        "state": state,
        "sealed_day_count": len(valid),
        "request_day_count": sum(row["state"] == "REQUESTS_SEALED" for row in valid),
        "missing_day_count": sum(
            row["state"] == "MISSING_SOURCE_DEADLINE" for row in valid
        ),
        "allocated_cell_count": 90,
        "response_sealed_count": response_count,
        "execution_failure_cell_count": execution_failure_count,
        "diagnostic_import_cell_count": diagnostic_import_count,
        "missing_response_cell_count": missing_response_count,
        "missing_source_cell_count": missing_source_cells,
        "fixed_cell_count": fixed_cells,
        "next_ordinal": next_ordinal if next_ordinal <= 30 else None,
        "promotion_eligible": False,
        "live_permission": False,
        "evidence_tier": "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC",
        "truth_day_score_count": len(day_scores),
        "phase_score_present": phase_score is not None,
        "validity_registry": validity_status_artifact(args.run_dir),
    }
    if next_ordinal <= 30:
        result["next_schedule"] = precommit["schedule"][next_ordinal - 1]
    return result


def _validity_status(args: argparse.Namespace) -> dict[str, Any]:
    with _run_lock(args.run_dir):
        return validity_status_artifact(args.run_dir)


def _invalidate(args: argparse.Namespace) -> dict[str, Any]:
    with _run_lock(args.run_dir):
        if _superseded_status(args.run_dir) is not None:
            raise RuntimeError("AI forward run is terminally superseded")
        append_invalidation(
            args.run_dir,
            logical_id=args.logical_id,
            reason_code=args.reason_code,
            evidence_sha256=args.evidence_sha256,
            invalidated_at_utc=_now(),
        )
        return validity_status_artifact(args.run_dir)


def _get_with_retry(
    client: OandaReadOnlyClient,
    path: str,
    query: dict[str, str],
    *,
    attempts: int,
) -> dict[str, Any]:
    error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return client.get_json(path, query)
        except Exception as exc:  # noqa: BLE001 - bounded evidence fetch retry.
            error = exc
            if attempt < attempts:
                time.sleep(float(attempt))
    assert error is not None
    raise error


@contextmanager
def _run_lock(run_dir: Path) -> Iterator[None]:
    if run_dir.is_symlink():
        raise RuntimeError("AI forward run directory cannot be a symlink")
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".ai-forward.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another AI forward operator owns the run") from exc
        yield
    finally:
        os.close(descriptor)


def _strict_json(path: Path) -> dict[str, Any]:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise RuntimeError(f"JSON evidence path contains a symlink: {path}")
    if not absolute.is_file():
        raise RuntimeError(f"JSON evidence path is not a regular file: {path}")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON number: {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_json_new_or_same(
    path: Path, value: Mapping[str, Any], *, root: Path
) -> None:
    data = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    root = root.absolute()
    path = path.absolute()
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("AI artifact root is absent or a symlink")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("AI artifact path escapes its run directory") from exc
    if not relative.parts or ".." in relative.parts:
        raise RuntimeError("AI artifact relative path is unsafe")
    parent = root
    for part in relative.parent.parts:
        parent = parent / part
        if parent.exists() or parent.is_symlink():
            if parent.is_symlink() or not parent.is_dir():
                raise RuntimeError("AI artifact parent is unsafe")
        else:
            parent.mkdir(mode=0o700)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise RuntimeError(f"immutable AI artifact already differs: {path}")
        _clean_pending(path)
        return
    prefix = f".{path.name}.pending-"
    descriptor, pending_text = tempfile.mkstemp(prefix=prefix, dir=path.parent)
    pending = Path(pending_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(pending, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
                raise RuntimeError(
                    f"immutable AI artifact concurrently differs: {path}"
                )
        pending.unlink()
        _fsync_directory(path.parent)
        _clean_pending(path)
    except BaseException:
        try:
            pending.unlink()
        except OSError:
            pass
        raise


def _clean_pending(path: Path) -> None:
    for candidate in path.parent.glob(f".{path.name}.pending-*"):
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError("pending AI evidence artifact is unsafe")
        candidate.unlink()
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    precommit = subparsers.add_parser("precommit")
    precommit.add_argument("--spec", type=Path, required=True)
    precommit.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    precommit.add_argument("--run-dir", type=Path, required=True)
    precommit.set_defaults(handler=_precommit)

    start = subparsers.add_parser("start")
    start.add_argument("--run-dir", type=Path, required=True)
    start.set_defaults(handler=_start)

    collect = subparsers.add_parser("collect-day")
    collect.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    collect.add_argument("--run-dir", type=Path, required=True)
    collect.add_argument("--ordinal", type=int, required=True)
    collect.add_argument("--attempts", type=int, default=3, choices=range(1, 6))
    collect.set_defaults(handler=_collect_day)

    missing = subparsers.add_parser("seal-missing-day")
    missing.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    missing.add_argument("--run-dir", type=Path, required=True)
    missing.add_argument("--ordinal", type=int, required=True)
    missing.set_defaults(handler=_seal_missing)

    response = subparsers.add_parser("import-diagnostic-response")
    response.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    response.add_argument("--run-dir", type=Path, required=True)
    response.add_argument("--ordinal", type=int, required=True)
    response.add_argument("--cell-id", required=True)
    response.add_argument("--response-json", type=Path, required=True)
    response.set_defaults(handler=_seal_cell_response)

    missing_responses = subparsers.add_parser("seal-missing-responses")
    missing_responses.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    missing_responses.add_argument("--run-dir", type=Path, required=True)
    missing_responses.add_argument("--ordinal", type=int, required=True)
    missing_responses.set_defaults(handler=_seal_missing_responses)

    index = subparsers.add_parser("seal-phase-index")
    index.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    index.add_argument("--run-dir", type=Path, required=True)
    index.set_defaults(handler=_seal_phase_index)

    truth = subparsers.add_parser("collect-truth-day")
    truth.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    truth.add_argument("--run-dir", type=Path, required=True)
    truth.add_argument("--ordinal", type=int, required=True)
    truth.add_argument("--attempts", type=int, default=3, choices=range(1, 6))
    truth.set_defaults(handler=_collect_truth)

    phase_score = subparsers.add_parser("seal-phase-score")
    phase_score.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    phase_score.add_argument("--run-dir", type=Path, required=True)
    phase_score.set_defaults(handler=_seal_phase_score)

    status = subparsers.add_parser("status")
    status.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    status.add_argument("--run-dir", type=Path, required=True)
    status.set_defaults(handler=_status)

    validity = subparsers.add_parser("validity-status")
    validity.add_argument("--run-dir", type=Path, required=True)
    validity.set_defaults(handler=_validity_status)

    invalidate = subparsers.add_parser("invalidate-artifact")
    invalidate.add_argument("--run-dir", type=Path, required=True)
    invalidate.add_argument("--logical-id", required=True)
    invalidate.add_argument("--reason-code", required=True)
    invalidate.add_argument("--evidence-sha256", required=True)
    invalidate.set_defaults(handler=_invalidate)

    args = parser.parse_args()
    # Every artifact helper treats child paths as relative to one canonical
    # run root.  Resolve the CLI root once so a caller's relative `--run-dir`
    # cannot be prefixed a second time during validity-ledger registration.
    args.run_dir = args.run_dir.resolve()
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
