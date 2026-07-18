#!/usr/bin/env python3
"""Build a sealed, read-only DOJO edge/3x classification board."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from quant_rabbit.dojo_goal_board import (
    DojoGoalBoardError,
    build_goal_board,
    load_goal_board_input,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _exclusive_write_json(path: Path, value: object) -> None:
    """Publish a complete board once; an existing path is immutable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic on the same filesystem and fails
        # with EEXIST instead of replacing an earlier sealed board.
        os.link(temporary_path, path, follow_symlinks=False)
        temporary_path.unlink()
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify DOJO worker/AI evidence separately from monthly 3x "
            "distribution compatibility. This command grants no live permission."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.input.resolve() == args.output.resolve(strict=False):
            raise DojoGoalBoardError("input and output must be different paths")
        source = load_goal_board_input(args.input)
        board = build_goal_board(source, project_root=PROJECT_ROOT)
        _exclusive_write_json(args.output, board)
    except (DojoGoalBoardError, OSError) as exc:
        print(f"dojo-goal-board: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "BUILT",
                "edge_status": board["edge_status"],
                "goal_status": board["goal_status"],
                "output": str(args.output),
                "guarantee": False,
                "live_permission": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
