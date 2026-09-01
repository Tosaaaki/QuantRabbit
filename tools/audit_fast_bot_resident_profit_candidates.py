#!/usr/bin/env python3
"""Audit all commit-addressed fast-bot resident ledgers without broker writes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.fast_bot_profit_candidate_audit import (  # noqa: E402
    audit_resident_profit_candidates,
    render_audit_report,
)


DEFAULT_STATE_ROOT = (
    Path.home()
    / ".codex"
    / "state"
    / "quantrabbit"
    / "owner-forward-shadow-resident-v1"
)


def main() -> int:
    args = _parser().parse_args()
    audit = audit_resident_profit_candidates(args.state_root)
    if args.output is not None:
        _write_atomic(args.output, json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if args.report is not None:
        _write_atomic(args.report, render_audit_report(audit))
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0 if audit["status"] != "REJECT_SOURCE_INTEGRITY" else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def _write_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
