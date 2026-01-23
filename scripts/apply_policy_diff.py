#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from analytics.policy_apply import apply_policy_diff_to_paths
from analytics.policy_diff import normalize_policy_diff, validate_policy_diff
from analytics.policy_ledger import PolicyLedger


def _load_diff(path: Path | None) -> dict:
    if path is None:
        raw = sys.stdin.read()
    else:
        raw = path.read_text()
    return json.loads(raw)


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply a policy_diff JSON to the overlay.")
    ap.add_argument("--diff", default=None, help="Path to policy_diff.json (stdin if omitted)")
    ap.add_argument("--overlay-path", default="logs/policy_overlay.json")
    ap.add_argument("--history-dir", default="logs/policy_history")
    ap.add_argument("--latest-path", default="logs/policy_latest.json")
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    diff_path = Path(args.diff) if args.diff else None
    payload = _load_diff(diff_path)
    payload = normalize_policy_diff(payload, source=str(payload.get("source") or "manual"))
    errors = validate_policy_diff(payload)
    if errors:
        logging.error("[POLICY] invalid policy_diff: %s", ", ".join(errors))
        raise SystemExit(2)

    updated, changed, flags = apply_policy_diff_to_paths(
        payload,
        overlay_path=Path(args.overlay_path),
        history_dir=Path(args.history_dir),
        latest_path=Path(args.latest_path),
    )
    logging.info(
        "[POLICY] applied=%s reentry=%s tuning=%s overlay=%s",
        changed,
        flags.get("reentry"),
        flags.get("tuning"),
        args.overlay_path,
    )

    if not args.no_ledger:
        ledger = PolicyLedger()
        ledger.record(payload, status="applied")
        if changed:
            ledger.record(updated, status="applied_snapshot")


if __name__ == "__main__":
    main()
