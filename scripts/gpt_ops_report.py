#!/usr/bin/env python3
"""LLM-disabled ops report stub.

This script keeps the CLI surface for systemd/cron callers but does not call
any LLM providers. It writes a minimal JSON report and an optional no-change
policy diff.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from analytics.policy_diff import normalize_policy_diff


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description="Ops report (LLM disabled)")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--output", default="logs/gpt_ops_report.json")
    ap.add_argument("--policy", action="store_true")
    ap.add_argument("--policy-output", default="logs/policy_diff_ops.json")
    ap.add_argument("--apply-policy", action="store_true")
    ap.add_argument("--gpt", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args, _ = ap.parse_known_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    now = datetime.now(timezone.utc).isoformat()

    payload = {
        "generated_at": now,
        "llm_disabled": True,
        "hours": args.hours,
        "note": "LLM features removed; report is minimal.",
    }
    _write_json(Path(args.output), payload)
    logging.info("[OPS_REPORT] wrote %s", args.output)

    if args.policy or args.apply_policy:
        diff = normalize_policy_diff(
            {"no_change": True, "reason": "llm_disabled", "source": "ops_stub"},
            source="ops_stub",
        )
        _write_json(Path(args.policy_output), diff)
        logging.info("[OPS_POLICY] wrote %s", args.policy_output)
        if args.apply_policy:
            logging.info("[OPS_POLICY] apply requested but LLM is disabled; skipping.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
