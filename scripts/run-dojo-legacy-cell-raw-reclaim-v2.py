#!/usr/bin/env python3
"""Plan, authorize, reclaim, or restore generation-2 legacy DOJO raw."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.dojo_legacy_cell_raw_reclaim_v2 import (  # noqa: E402
    DojoLegacyCellRawReclaimV2Error,
    build_v2_attestation_body_candidate,
    build_v2_candidate_plan,
    enroll_v2_attestation_public_key,
    load_v2_plan,
    publish_v2_plan,
    reclaim_generation_2_raw,
    restore_raw_from_v2_plan,
    verify_signed_attestations,
)


def _lineage_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--prior-plan", type=Path, required=True)
    parser.add_argument("--prior-receipt", type=Path, required=True)
    parser.add_argument("--prior-remote-receipts-dir", type=Path, required=True)
    parser.add_argument("--expected-drive-parent-id", required=True)
    parser.add_argument("--zstd-bin", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="build a read-only V2 candidate")
    _lineage_options(verify)
    verify.add_argument("--attestation-authority-seal", type=Path, required=True)
    plan = commands.add_parser("plan", help="seal and publish a V2 plan")
    _lineage_options(plan)
    plan.add_argument("--attestation-authority-seal", type=Path, required=True)
    plan.add_argument("--confirm-public-key-sha256", required=True)
    enroll = commands.add_parser(
        "enroll-key", help="enroll one Ed25519 public key before Drive observation"
    )
    enroll.add_argument("--source-run", type=Path, required=True)
    enroll.add_argument("--attestation-public-key-file", type=Path, required=True)
    enroll.add_argument("--confirm-public-key-sha256", required=True)
    body = commands.add_parser(
        "build-attestation-body",
        help=(
            "validate files.get before/after, revisions.list, and independent "
            "current-revision readback evidence into one unsigned body"
        ),
    )
    body.add_argument("--plan", type=Path, required=True)
    body.add_argument("--observation-json", type=Path, required=True)
    reclaim = commands.add_parser(
        "reclaim", help="release raw authorized by signed Drive attestations"
    )
    reclaim.add_argument("--plan", type=Path, required=True)
    reclaim.add_argument("--attestations-dir", type=Path, required=True)
    reclaim.add_argument("--expected-plan-sha256", required=True)
    reclaim.add_argument("--expected-target-count", type=int, required=True)
    reclaim.add_argument("--expected-target-bytes", type=int, required=True)
    signed = commands.add_parser(
        "verify-attestations",
        help="verify the complete signed Drive set without releasing raw",
    )
    signed.add_argument("--plan", type=Path, required=True)
    signed.add_argument("--attestations-dir", type=Path, required=True)
    restore = commands.add_parser(
        "restore", help="restore sealed raw without overwrite"
    )
    restore.add_argument("--plan", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    restore.add_argument(
        "--scope", choices=("prior", "generation2", "all"), required=True
    )
    restore.add_argument("--expected-plan-sha256", required=True)
    return parser


def _public_key(path: Path | None) -> str | None:
    if path is None:
        return None
    if path.is_symlink() or not path.is_file():
        raise DojoLegacyCellRawReclaimV2Error("public-key file is unsafe")
    value = path.read_text(encoding="ascii").strip()
    if len(value) != 64:
        raise DojoLegacyCellRawReclaimV2Error(
            "public-key file must contain one 32-byte hex key"
        )
    return value


def _observation(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise DojoLegacyCellRawReclaimV2Error("observation JSON is unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoLegacyCellRawReclaimV2Error("observation JSON is invalid") from exc
    if not isinstance(value, dict):
        raise DojoLegacyCellRawReclaimV2Error(
            "observation JSON must contain one object"
        )
    return value


def _candidate(args: argparse.Namespace) -> dict:
    return build_v2_candidate_plan(
        source_run=args.source_run,
        archive_root=args.archive_root,
        prior_plan_path=args.prior_plan,
        prior_receipt_path=args.prior_receipt,
        prior_remote_receipts_dir=args.prior_remote_receipts_dir,
        expected_drive_parent_id=args.expected_drive_parent_id,
        zstd_bin=args.zstd_bin,
        attestation_authority_seal_path=args.attestation_authority_seal,
    )


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "enroll-key":
            result = enroll_v2_attestation_public_key(
                source_run=args.source_run,
                attestation_public_key_hex=_public_key(
                    args.attestation_public_key_file
                ),
                expected_public_key_sha256=args.confirm_public_key_sha256,
            )
        elif args.command == "verify":
            result: object = _candidate(args)
        elif args.command == "plan":
            candidate = _candidate(args)
            fingerprint = candidate["attestation_authority"]["public_key_sha256"]
            if args.confirm_public_key_sha256 != fingerprint:
                raise DojoLegacyCellRawReclaimV2Error(
                    "public-key fingerprint confirmation differs"
                )
            path = publish_v2_plan(candidate)
            result = {
                "status": "V2_PLAN_PUBLISHED",
                "path": str(path),
                "plan": candidate,
            }
        elif args.command == "build-attestation-body":
            result = build_v2_attestation_body_candidate(
                plan=load_v2_plan(args.plan),
                observation=_observation(args.observation_json),
            )
        elif args.command == "reclaim":
            result = reclaim_generation_2_raw(
                plan_path=args.plan,
                attestations_dir=args.attestations_dir,
                expected_plan_sha256=args.expected_plan_sha256,
                expected_target_count=args.expected_target_count,
                expected_target_bytes=args.expected_target_bytes,
            )
        elif args.command == "verify-attestations":
            plan = load_v2_plan(args.plan)
            attestations = verify_signed_attestations(
                plan=plan, attestations_dir=args.attestations_dir
            )
            result = {
                "status": "SIGNED_ATTESTATION_SET_VERIFIED_NOT_EXECUTED",
                "plan_sha256": plan["reclaim_plan_sha256"],
                "attestation_count": len(attestations),
                "attestation_sha256s": [
                    row["attestation_sha256"] for row in attestations
                ],
                "source_deletion_allowed": False,
                "live_permission": False,
                "order_authority": "NONE",
            }
        else:
            result = restore_raw_from_v2_plan(
                plan_path=args.plan,
                destination=args.destination,
                scope=args.scope,
                expected_plan_sha256=args.expected_plan_sha256,
            )
    except (DojoLegacyCellRawReclaimV2Error, OSError) as exc:
        print(
            json.dumps({"status": "REJECTED", "error": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
