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
    precommit = validate_precommit(_strict_json(run_dir / "precommit.json"))
    start = validate_start_receipt(_strict_json(run_dir / "start.json"), precommit)
    return precommit, start


def _previous_day(run_dir: Path, ordinal: int) -> dict[str, Any] | None:
    if ordinal == 1:
        return None
    return _strict_json(run_dir / "days" / f"day-{ordinal - 1:03d}.json")


def _precommit(args: argparse.Namespace) -> dict[str, Any]:
    with _run_lock(args.run_dir):
        registry = assert_locked_preregistration(_strict_json(args.registry))
        prompt_texts = {
            row["variant_id"]: (REPO / row["prompt_path"]).read_text(
                encoding="utf-8"
            )
            for row in registry["variants"]
        }
        artifact = build_precommit(
            registry,
            prompt_texts,
            _strict_json(args.spec),
            now_utc=_now(),
        )
        _write_json_new_or_same(
            args.run_dir / "precommit.json", artifact, root=args.run_dir
        )
        return artifact


def _start(args: argparse.Namespace) -> dict[str, Any]:
    with _run_lock(args.run_dir):
        precommit = validate_precommit(_strict_json(args.run_dir / "precommit.json"))
        receipt = build_start_receipt(precommit, now_utc=_now())
        _write_json_new_or_same(
            args.run_dir / "start.json", receipt, root=args.run_dir
        )
        return receipt


def _collect_day(args: argparse.Namespace) -> dict[str, Any]:
    with _run_lock(args.run_dir):
        registry = assert_locked_preregistration(_strict_json(args.registry))
        precommit, start = _load_parents(args.run_dir)
        days = _validated_days(args.run_dir, registry, precommit, start)
        if args.ordinal < 1 or args.ordinal > len(days) + 1:
            raise RuntimeError("AI source collection ordinal is not the next chain day")
        previous = days[args.ordinal - 2] if args.ordinal > 1 else None
        day_path = args.run_dir / "days" / f"day-{args.ordinal:03d}.json"
        capture_path = (
            args.run_dir / "source-captures" / f"day-{args.ordinal:03d}.json"
        )
        request_path = (
            args.run_dir / "source-requests" / f"day-{args.ordinal:03d}.json"
        )
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
            return day
        if capture_path.is_file():
            capture = validate_source_capture(
                _strict_json(capture_path), precommit, start, previous
            )
            seal = build_day_requests_from_capture(
                registry,
                precommit,
                start,
                previous,
                capture,
            )
            _write_json_new_or_same(day_path, seal, root=args.run_dir)
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
        seal = build_day_requests_from_capture(
            registry,
            precommit,
            start,
            previous,
            capture,
        )
        _write_json_new_or_same(day_path, seal, root=args.run_dir)
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
        capture_path = (
            args.run_dir / "source-captures" / f"day-{args.ordinal:03d}.json"
        )
        failed_capture_sha: str | None = None
        if capture_path.is_file():
            capture = validate_source_capture(
                _strict_json(capture_path), precommit, start, previous
            )
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
                terminal["state"] != "RESPONSE_SEALED"
                or terminal["response_receipt"]["response"] != response
            ):
                raise RuntimeError("immutable cell terminal already differs")
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
                "contract": "QR_DOJO_AI_FORWARD_MISSING_RESPONSE_BATCH_V2",
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
                existing += 1
                continue
            terminal = build_cell_response_failure(
                precommit,
                day,
                cell_id=cell_id,
                now_utc=now,
            )
            _write_json_new_or_same(path, terminal, root=args.run_dir)
            created += 1
        return {
            "contract": "QR_DOJO_AI_FORWARD_MISSING_RESPONSE_BATCH_V2",
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
        elif seal["state"] == "REQUESTS_SEALED" or capture_path.exists():
            raise RuntimeError("AI day and persisted capture are not joined")
        valid.append(seal)
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
            terminals.append(terminal)
    response_root = run_dir / "responses"
    if response_root.is_symlink():
        raise RuntimeError("AI response directory cannot be a symlink")
    actual_paths = set(response_root.rglob("*.json")) if response_root.exists() else set()
    unexpected = actual_paths - expected_paths
    if unexpected:
        raise RuntimeError("unexpected AI cell terminal artifact is present")
    return terminals


def _seal_phase_index(args: argparse.Namespace) -> dict[str, Any]:
    with _run_lock(args.run_dir):
        registry = assert_locked_preregistration(_strict_json(args.registry))
        precommit, start = _load_parents(args.run_dir)
        days = _validated_days(args.run_dir, registry, precommit, start)
        terminals = _validated_terminals(
            args.run_dir, registry, precommit, start, days
        )
        path = args.run_dir / "phase-index.json"
        if path.is_file():
            return validate_phase_index(
                _strict_json(path), registry, precommit, start, days, terminals
            )
        index = build_phase_index(
            registry,
            precommit,
            start,
            days,
            terminals,
            now_utc=_now(),
        )
        _write_json_new_or_same(path, index, root=args.run_dir)
        return index


def _status(args: argparse.Namespace) -> dict[str, Any]:
    registry = assert_locked_preregistration(_strict_json(args.registry))
    precommit, start = _load_parents(args.run_dir)
    valid = _validated_days(args.run_dir, registry, precommit, start)
    terminals = _validated_terminals(
        args.run_dir, registry, precommit, start, valid
    )
    next_ordinal = len(valid) + 1
    response_count = sum(row["state"] == "RESPONSE_SEALED" for row in terminals)
    missing_response_count = sum(
        row["state"] == "MISSING_RESPONSE_DEADLINE" for row in terminals
    )
    missing_source_cells = 3 * sum(
        row["state"] == "MISSING_SOURCE_DEADLINE" for row in valid
    )
    fixed_cells = response_count + missing_response_count + missing_source_cells
    if next_ordinal <= 30:
        state = "COLLECTING_SOURCE"
    elif fixed_cells < 90:
        state = "COLLECTING_RESPONSES"
    else:
        state = "RESPONSES_FIXED_AWAITING_MARKET_TRUTH"
    result: dict[str, Any] = {
        "contract": "QR_DOJO_AI_FORWARD_STATUS_V2",
        "experiment_id": precommit["experiment_id"],
        "state": state,
        "sealed_day_count": len(valid),
        "request_day_count": sum(row["state"] == "REQUESTS_SEALED" for row in valid),
        "missing_day_count": sum(
            row["state"] == "MISSING_SOURCE_DEADLINE" for row in valid
        ),
        "allocated_cell_count": 90,
        "response_sealed_count": response_count,
        "missing_response_cell_count": missing_response_count,
        "missing_source_cell_count": missing_source_cells,
        "fixed_cell_count": fixed_cells,
        "next_ordinal": next_ordinal if next_ordinal <= 30 else None,
        "promotion_eligible": False,
        "live_permission": False,
        "evidence_tier": "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC",
    }
    if next_ordinal <= 30:
        result["next_schedule"] = precommit["schedule"][next_ordinal - 1]
    return result


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
                raise RuntimeError(f"immutable AI artifact concurrently differs: {path}")
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

    response = subparsers.add_parser("seal-response")
    response.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    response.add_argument("--run-dir", type=Path, required=True)
    response.add_argument("--ordinal", type=int, required=True)
    response.add_argument("--cell-id", required=True)
    response.add_argument("--response-json", type=Path, required=True)
    response.set_defaults(handler=_seal_cell_response)

    missing_responses = subparsers.add_parser("seal-missing-responses")
    missing_responses.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY
    )
    missing_responses.add_argument("--run-dir", type=Path, required=True)
    missing_responses.add_argument("--ordinal", type=int, required=True)
    missing_responses.set_defaults(handler=_seal_missing_responses)

    index = subparsers.add_parser("seal-phase-index")
    index.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    index.add_argument("--run-dir", type=Path, required=True)
    index.set_defaults(handler=_seal_phase_index)

    status = subparsers.add_parser("status")
    status.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    status.add_argument("--run-dir", type=Path, required=True)
    status.set_defaults(handler=_status)

    args = parser.parse_args()
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
