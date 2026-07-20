#!/usr/bin/env python3
"""Prepare or run one broker-free DOJO long-horizon economic stream job."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.dojo_long_horizon_economic_runner import (
    BUILTIN_NO_INTENT_RUNTIME_BINDING_SHA256,
    DojoLongHorizonEconomicRunnerError,
    build_month_source_slice_receipt,
    builtin_no_intent_runtime_factory,
    run_long_horizon_economic_job,
)
from quant_rabbit.dojo_long_horizon_plan import canonical_sha256


MAX_INPUT_BYTES = 64 * 1024 * 1024


def _read_json(path: Path) -> dict[str, Any]:
    state = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(state.st_mode) or state.st_size <= 0 or state.st_size > MAX_INPUT_BYTES:
        raise DojoLongHorizonEconomicRunnerError(
            f"input is not a bounded regular file: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        raw = handle.read(MAX_INPUT_BYTES + 1)
    if len(raw) != state.st_size:
        raise DojoLongHorizonEconomicRunnerError(f"input changed while read: {path}")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoLongHorizonEconomicRunnerError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DojoLongHorizonEconomicRunnerError(f"input must be an object: {path}")
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    digest = canonical_sha256(value)
    temporary = path.with_name(f".{path.name}.{digest}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            if handle.write(payload) != len(payload):
                raise DojoLongHorizonEconomicRunnerError(
                    "output write was incomplete"
                )
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and cannot overwrite an existing
        # immutable final name.  ENOSPC/short-write can strand only a temp file.
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    source = sub.add_parser("seal-source-slice")
    source.add_argument("--runner-handoff", type=Path, required=True)
    source.add_argument("--source-root", type=Path, required=True)
    source.add_argument("--source-manifest", type=Path, required=True)
    source.add_argument("--relative-path", required=True)
    source.add_argument("--output", type=Path, required=True)

    run = sub.add_parser("run")
    run.add_argument("--runner-handoff", type=Path, required=True)
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--source-manifest", type=Path, required=True)
    run.add_argument("--source-slice-receipt", type=Path, required=True)
    run.add_argument("--worker-catalog", type=Path, required=True)
    run.add_argument("--coordinate-runtimes", type=Path, required=True)
    run.add_argument(
        "--builtin-no-intent-only",
        action="store_true",
        help="Run the capability-closed built-in HOLD baseline; external Python is forbidden.",
    )
    run.add_argument("--carry-states", type=Path)
    run.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        handoff = _read_json(args.runner_handoff)
        if args.command == "seal-source-slice":
            job = handoff.get("job")
            if not isinstance(job, dict):
                raise DojoLongHorizonEconomicRunnerError(
                    "runner handoff has no job object"
                )
            result = build_month_source_slice_receipt(
                source_root=args.source_root,
                relative_path=args.relative_path,
                job=job,
                source_manifest=_read_json(args.source_manifest),
            )
        else:
            catalog_doc = _read_json(args.worker_catalog)
            runtime_doc = _read_json(args.coordinate_runtimes)
            catalog = catalog_doc.get("worker_catalog")
            runtimes = runtime_doc.get("coordinate_runtimes")
            if not isinstance(catalog, list) or not isinstance(runtimes, dict):
                raise DojoLongHorizonEconomicRunnerError(
                    "catalog/runtime wrapper schema is invalid"
                )
            carry: Mapping[str, Mapping[str, Any]] = {}
            if args.carry_states is not None:
                carry_doc = _read_json(args.carry_states)
                raw_carry = carry_doc.get("economic_carry_states_by_slot")
                if not isinstance(raw_carry, dict):
                    raise DojoLongHorizonEconomicRunnerError(
                        "carry wrapper schema is invalid"
                    )
                carry = raw_carry
            if not args.builtin_no_intent_only:
                raise DojoLongHorizonEconomicRunnerError(
                    "CLI permits only --builtin-no-intent-only; arbitrary worker code "
                    "must not be loaded in the evidence process"
                )
            result = run_long_horizon_economic_job(
                runner_handoff=handoff,
                plan=_read_json(args.plan),
                source_root=args.source_root,
                source_manifest=_read_json(args.source_manifest),
                source_slice_receipt=_read_json(args.source_slice_receipt),
                worker_catalog=catalog,
                coordinate_runtimes=runtimes,
                worker_runtime_factory=builtin_no_intent_runtime_factory,
                worker_runtime_binding_sha256=BUILTIN_NO_INTENT_RUNTIME_BINDING_SHA256,
                carry_states_by_slot=carry,
            )
        _write_exclusive(args.output, result)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "sha256": result.get("economic_job_result_sha256")
                    or result.get("source_slice_receipt_sha256"),
                    "live_permission": False,
                    "broker_mutation_allowed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    except (DojoLongHorizonEconomicRunnerError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
