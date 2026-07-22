#!/usr/bin/env python3
"""Validate local readback claims into a non-authoritative candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_rabbit.dojo_historical_raw_reclaim import (
    create_historical_job_remote_readback_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate locally supplied Google Drive metadata claims and downloaded "
            "bytes, then write one content-addressed CANDIDATE_ONLY record. This "
            "command never attests provider provenance and cannot authorize reclaim."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--archive-receipt", type=Path, required=True)
    parser.add_argument("--evidence-packet", type=Path, required=True)
    parser.add_argument("--expected-drive-parent-id", required=True)
    parser.add_argument("--zstd-bin", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = create_historical_job_remote_readback_receipt(
        run_root=args.run_root,
        archive_receipt_path=args.archive_receipt,
        evidence_packet_path=args.evidence_packet,
        expected_drive_parent_id=args.expected_drive_parent_id,
        zstd_bin=args.zstd_bin,
    )
    archive_root = args.archive_receipt.resolve(strict=True).parent.parent
    receipt_path = (
        archive_root
        / "remote-candidates"
        / (
            f"candidate-job-{receipt['job_sha256']}-{receipt['manifest_sha256']}-"
            f"{receipt['candidate_sha256']}.json"
        )
    )
    result = {
        "status": receipt["status"],
        "receipt_path": str(receipt_path),
        "candidate_sha256": receipt["candidate_sha256"],
        "remote_verified": receipt["remote_verified"],
        "raw_reclaim_eligible": receipt["raw_reclaim_eligible"],
        "trusted_provider_attestation_present": receipt[
            "trusted_provider_attestation_present"
        ],
        "source_deleted": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
