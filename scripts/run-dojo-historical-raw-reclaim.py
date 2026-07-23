#!/usr/bin/env python3
"""Verify or execute one externally attested historical DOJO raw reclaim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_rabbit.dojo_historical_raw_reclaim import (
    enroll_historical_job_attestation_public_key,
    historical_global_heavy_lock_path,
    publish_historical_job_signed_remote_readback_receipt,
    reclaim_historical_job_raw,
    restore_historical_job_raw,
    verify_historical_job_raw_reclaim,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify, reclaim, or restore exact historical run-root raw files. "
            "Archive deletion and DriveFS local-cache eviction are separate "
            "explicit operations and are never performed by this command."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "reclaim"):
        command = commands.add_parser(name)
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--archive-receipt", type=Path, required=True)
        command.add_argument("--remote-receipt", type=Path, required=True)
        command.add_argument("--expected-drive-parent-id", required=True)
        command.add_argument(
            "--attestation-authority-seal",
            type=Path,
            required=True,
            help="pre-observation append-only Ed25519 public-key seal",
        )
        command.add_argument(
            "--zstd-bin",
            required=True,
            help="absolute path to the sealed zstd executable",
        )
        command.add_argument(
            "--global-heavy-lock",
            type=Path,
            help=(
                "machine-wide heavy-operation lease; when omitted, use the path "
                "from the generation's sealed run control"
            ),
        )
        if name == "reclaim":
            command.add_argument(
                "--confirm-reclaim",
                action="store_true",
                help="required acknowledgement for allowlisted raw retirement",
            )
            command.add_argument("--confirm-plan-sha256", required=True)
            command.add_argument("--confirm-target-count", type=int, required=True)
            command.add_argument("--confirm-target-bytes", type=int, required=True)
    enroll = commands.add_parser("enroll-key")
    enroll.add_argument("--run-root", type=Path, required=True)
    enroll.add_argument("--archive-receipt", type=Path, required=True)
    enroll.add_argument("--expected-drive-parent-id", required=True)
    enroll.add_argument("--attestation-public-key-hex", required=True)
    enroll.add_argument("--zstd-bin", required=True)
    publish = commands.add_parser("publish-signed")
    publish.add_argument("--run-root", type=Path, required=True)
    publish.add_argument("--archive-receipt", type=Path, required=True)
    publish.add_argument("--signed-attestation", type=Path, required=True)
    publish.add_argument("--expected-drive-parent-id", required=True)
    publish.add_argument("--attestation-authority-seal", type=Path, required=True)
    publish.add_argument("--zstd-bin", required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("--run-root", type=Path, required=True)
    restore.add_argument("--archive-receipt", type=Path, required=True)
    restore.add_argument("--reclaim-plan", type=Path, required=True)
    restore.add_argument("--reclaim-receipt", type=Path, required=True)
    restore.add_argument(
        "--zstd-bin",
        required=True,
        help="absolute path to the recovery zstd executable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "enroll-key":
        result = enroll_historical_job_attestation_public_key(
            run_root=args.run_root,
            archive_receipt_path=args.archive_receipt,
            expected_drive_parent_id=args.expected_drive_parent_id,
            attestation_public_key_hex=args.attestation_public_key_hex,
            zstd_bin=args.zstd_bin,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "publish-signed":
        result = publish_historical_job_signed_remote_readback_receipt(
            run_root=args.run_root,
            archive_receipt_path=args.archive_receipt,
            signed_attestation_path=args.signed_attestation,
            expected_drive_parent_id=args.expected_drive_parent_id,
            attestation_authority_seal_path=args.attestation_authority_seal,
            zstd_bin=args.zstd_bin,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "restore":
        result = restore_historical_job_raw(
            run_root=args.run_root,
            archive_receipt_path=args.archive_receipt,
            reclaim_plan_path=args.reclaim_plan,
            reclaim_receipt_path=args.reclaim_receipt,
            zstd_bin=args.zstd_bin,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0

    common: dict[str, Any] = {
        "run_root": args.run_root,
        "archive_receipt_path": args.archive_receipt,
        "remote_receipt_path": args.remote_receipt,
        "expected_drive_parent_id": args.expected_drive_parent_id,
        "attestation_authority_seal_path": args.attestation_authority_seal,
        "zstd_bin": args.zstd_bin,
        "global_heavy_lock_path": (
            args.global_heavy_lock
            if args.global_heavy_lock is not None
            else historical_global_heavy_lock_path(run_root=args.run_root)
        ),
    }
    if args.command == "verify":
        result = verify_historical_job_raw_reclaim(**common)
    else:
        if not args.confirm_reclaim:
            raise SystemExit("reclaim requires --confirm-reclaim")
        result = reclaim_historical_job_raw(
            **common,
            confirmed_plan_sha256=args.confirm_plan_sha256,
            confirmed_target_count=args.confirm_target_count,
            confirmed_target_bytes=args.confirm_target_bytes,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
