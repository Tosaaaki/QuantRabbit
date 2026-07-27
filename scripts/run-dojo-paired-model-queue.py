#!/usr/bin/env python3
"""Operate the content-addressed Codex decision queue for paired DOJO replay."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from quant_rabbit.dojo_paired_model_queue import (
    current_ready_packet,
    emit_next_ready,
    initialize_queue,
    queue_status,
    seal_model_response,
    submit_model_response,
    verify_queue,
)


def _read(path: Path) -> dict[str, Any]:
    with path.resolve(strict=True).open("rb") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _print(value: dict[str, Any]) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _init(args: argparse.Namespace) -> int:
    result_values = [_read(path) for path in sorted(args.results_dir.glob("*.json"))]
    _print(
        initialize_queue(
            queue_dir=args.queue_dir,
            source_plan=_read(args.source_plan),
            result_values=result_values,
        )
    )
    return 0


def _status(args: argparse.Namespace) -> int:
    _print(queue_status(args.queue_dir))
    return 0


def _emit(args: argparse.Namespace) -> int:
    _print(emit_next_ready(args.queue_dir))
    return 0


def _show_ready(args: argparse.Namespace) -> int:
    _print(current_ready_packet(args.queue_dir))
    return 0


def _seal_response(args: argparse.Namespace) -> int:
    response = seal_model_response(
        packet=current_ready_packet(args.queue_dir),
        action=args.action,
        reason_ids=args.reason_id,
        provider_model=args.provider_model,
        provider_execution_id=args.provider_execution_id,
    )
    _write_exclusive(args.output, response)
    _print(response)
    return 0


def _submit(args: argparse.Namespace) -> int:
    _print(
        submit_model_response(
            queue_dir=args.queue_dir,
            response_value=_read(args.response),
        )
    )
    return 0


def _verify(args: argparse.Namespace) -> int:
    _print(verify_queue(args.queue_dir))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--source-plan", type=Path, required=True)
    init.add_argument("--results-dir", type=Path, required=True)
    init.add_argument("--queue-dir", type=Path, required=True)
    init.set_defaults(handler=_init)
    status = subparsers.add_parser("status")
    status.add_argument("--queue-dir", type=Path, required=True)
    status.set_defaults(handler=_status)
    emit = subparsers.add_parser("emit-next")
    emit.add_argument("--queue-dir", type=Path, required=True)
    emit.set_defaults(handler=_emit)
    ready = subparsers.add_parser("show-ready")
    ready.add_argument("--queue-dir", type=Path, required=True)
    ready.set_defaults(handler=_show_ready)
    seal = subparsers.add_parser("seal-response")
    seal.add_argument("--queue-dir", type=Path, required=True)
    seal.add_argument(
        "--action",
        choices=tuple(
            sorted(
                {
                    "HOLD",
                    "PAUSE_NEW_ENTRIES",
                    "RESUME",
                    "REDUCE_LONG",
                    "REDUCE_SHORT",
                    "PARTIAL_CLOSE",
                    "CLOSE_RISKY",
                    "CLOSE_ALL",
                    "BLOCK_LONG_ENTRIES",
                    "BLOCK_SHORT_ENTRIES",
                }
            )
        ),
        required=True,
    )
    seal.add_argument("--reason-id", action="append", required=True)
    seal.add_argument("--provider-model", required=True)
    seal.add_argument("--provider-execution-id", required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.set_defaults(handler=_seal_response)
    submit = subparsers.add_parser("submit-response")
    submit.add_argument("--queue-dir", type=Path, required=True)
    submit.add_argument("--response", type=Path, required=True)
    submit.set_defaults(handler=_submit)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--queue-dir", type=Path, required=True)
    verify.set_defaults(handler=_verify)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
