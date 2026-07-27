#!/usr/bin/env python3
"""Operate the no-credential fresh Codex task handoff for DOJO supervision."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.dojo_fresh_model_handoff import (
    build_paper_source_packet_from_rooms,
    compile_snapshot,
    current_ready_packet,
    handoff_status,
    initialize_handoff,
    seal_model_response,
    submit_model_response,
    verify_handoff,
)
from quant_rabbit.dojo_paired_inventory_counterfactual import ACTION_IDS


def _read(path: Path) -> dict[str, Any]:
    with path.resolve(strict=True).open("rb") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_event_list(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    value = _read(path)
    events = value.get("events")
    if not isinstance(events, list) or any(
        not isinstance(item, dict) for item in events
    ):
        raise ValueError("recent event file must contain an events array")
    return events


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
    _print(initialize_handoff(args.root))
    return 0


def _compile(args: argparse.Namespace) -> int:
    _print(
        compile_snapshot(
            root=args.root,
            source_packet=_read(args.source_packet),
            recent_events=_read_event_list(args.recent_events),
            risk_signals=args.risk_signal,
            major_event_ids=args.major_event,
        )
    )
    return 0


def _compile_rooms(args: argparse.Namespace) -> int:
    now = (
        datetime.fromisoformat(args.now_utc.replace("Z", "+00:00"))
        if args.now_utc
        else datetime.now(timezone.utc)
    )
    source_packet, detected_risk_signals = build_paper_source_packet_from_rooms(
        rooms_root=args.rooms_root,
        now_utc=now,
    )
    _print(
        compile_snapshot(
            root=args.root,
            source_packet=source_packet,
            recent_events=_read_event_list(args.recent_events),
            risk_signals=sorted(set(detected_risk_signals) | set(args.risk_signal)),
            major_event_ids=args.major_event,
        )
    )
    return 0


def _show_ready(args: argparse.Namespace) -> int:
    _print(current_ready_packet(args.root))
    return 0


def _seal(args: argparse.Namespace) -> int:
    response = seal_model_response(
        packet=current_ready_packet(args.root),
        action=args.action,
        reason_ids=args.reason_id,
        next_story_content=_read(args.next_story),
        provider_model=args.provider_model,
        provider_execution_id=args.provider_execution_id,
    )
    _write_exclusive(args.output, response)
    _print(response)
    return 0


def _submit(args: argparse.Namespace) -> int:
    _print(
        submit_model_response(
            root=args.root,
            response_value=_read(args.response),
        )
    )
    return 0


def _status(args: argparse.Namespace) -> int:
    _print(handoff_status(args.root))
    return 0


def _verify(args: argparse.Namespace) -> int:
    _print(verify_handoff(args.root))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--root", type=Path, required=True)
    init.set_defaults(handler=_init)

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--root", type=Path, required=True)
    compile_parser.add_argument("--source-packet", type=Path, required=True)
    compile_parser.add_argument("--recent-events", type=Path)
    compile_parser.add_argument("--risk-signal", action="append", default=[])
    compile_parser.add_argument("--major-event", action="append", default=[])
    compile_parser.set_defaults(handler=_compile)

    rooms = subparsers.add_parser("compile-rooms")
    rooms.add_argument("--root", type=Path, required=True)
    rooms.add_argument("--rooms-root", type=Path, required=True)
    rooms.add_argument("--now-utc")
    rooms.add_argument("--recent-events", type=Path)
    rooms.add_argument("--risk-signal", action="append", default=[])
    rooms.add_argument("--major-event", action="append", default=[])
    rooms.set_defaults(handler=_compile_rooms)

    ready = subparsers.add_parser("show-ready")
    ready.add_argument("--root", type=Path, required=True)
    ready.set_defaults(handler=_show_ready)

    seal = subparsers.add_parser("seal-response")
    seal.add_argument("--root", type=Path, required=True)
    seal.add_argument("--action", choices=tuple(sorted(ACTION_IDS)), required=True)
    seal.add_argument("--reason-id", action="append", required=True)
    seal.add_argument("--next-story", type=Path, required=True)
    seal.add_argument("--provider-model", required=True)
    seal.add_argument("--provider-execution-id", required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.set_defaults(handler=_seal)

    submit = subparsers.add_parser("submit-response")
    submit.add_argument("--root", type=Path, required=True)
    submit.add_argument("--response", type=Path, required=True)
    submit.set_defaults(handler=_submit)

    status = subparsers.add_parser("status")
    status.add_argument("--root", type=Path, required=True)
    status.set_defaults(handler=_status)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.set_defaults(handler=_verify)

    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
