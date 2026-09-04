#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_rabbit.policy_snapshot import (
    PolicyBinding,
    PolicySnapshotError,
    load_and_verify_policy_snapshot,
    write_sealed_policy_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh or verify the local sealed AI-trader policy snapshot."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--input", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--snapshot", type=Path, required=True)
    verify.add_argument("--project-key", required=True)
    verify.add_argument("--broker-account-id", required=True)
    verify.add_argument("--environment", required=True)
    verify.add_argument("--revocation-epoch", required=True, type=int)
    verify.add_argument("--required-source-page", action="append", default=[])
    args = parser.parse_args()
    try:
        if args.command == "seal":
            raw = json.loads(args.input.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise PolicySnapshotError("POLICY_SNAPSHOT_INVALID", "input must be an object")
            sealed = write_sealed_policy_snapshot(args.output, raw)
            result = {
                "status": "SEALED",
                "snapshot_path": str(args.output),
                "snapshot_sha256": sealed["snapshot_sha256"],
            }
        else:
            verified = load_and_verify_policy_snapshot(
                args.snapshot,
                binding=PolicyBinding(
                    project_key=args.project_key,
                    broker_account_id=args.broker_account_id,
                    environment=args.environment,
                    revocation_epoch=args.revocation_epoch,
                ),
                required_source_pages=args.required_source_page,
            )
            result = {
                "status": "VERIFIED",
                "snapshot_path": str(args.snapshot),
                "policy_version": verified["policy_version"],
                "snapshot_sha256": verified["snapshot_sha256"],
                "expires_at_utc": verified["expires_at_utc"],
            }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, json.JSONDecodeError, PolicySnapshotError) as exc:
        code = exc.code if isinstance(exc, PolicySnapshotError) else "POLICY_SNAPSHOT_IO_ERROR"
        print(json.dumps({"status": "BLOCKED", "code": code, "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
