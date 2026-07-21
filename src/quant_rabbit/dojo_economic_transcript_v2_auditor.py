"""Independent compact-segment auditor for one terminal DOJO account chain.

This module is deliberately executable in a separate process.  It verifies the
entire compact V3 segment chain, restores the terminal reducer checkpoint, and derives
the terminal portfolio result without trusting the producing runner.  It has no
broker, model, worker, allocation, promotion, or live authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_economic_transcript_v2 import (
    verify_economic_segment,
    verify_economic_segment_chain,
)
from quant_rabbit.dojo_portfolio_replay_reducer import (
    MONTH_END_FLAT_SETTLEMENT,
    MONTH_END_MTM_WITH_STATE_HANDOFF,
    PortfolioReplaySession,
    canonical_portfolio_sha256,
    validate_portfolio_replay_result,
)


V2_REEXECUTION_ATTESTATION_CONTRACT: Final = (
    "QR_DOJO_ECONOMIC_TRANSCRIPT_SEGMENT_REEXECUTION_ATTESTATION_V3"
)
SCHEMA_VERSION: Final = 3
MAX_SEGMENT_PATHS: Final = 65_536
MAX_SEGMENT_FILE_BYTES: Final = 512 * 1024 * 1024
GENESIS_SHA256: Final = "0" * 64


class DojoEconomicV2AuditorError(ValueError):
    """A compact transcript cannot be independently reduced as one chain."""


def _file_digest(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size > MAX_SEGMENT_FILE_BYTES
        ):
            raise DojoEconomicV2AuditorError(
                "segment must be a bounded single-link regular file"
            )
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            while block := handle.read(1024 * 1024):
                digest.update(block)
            after = os.fstat(handle.fileno())
        current = path.stat(follow_symlinks=False)
        identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ) != identity:
            raise DojoEconomicV2AuditorError(
                "segment changed during file-digest verification"
            )
        return digest.hexdigest(), opened.st_size
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def reexecute_v2_economic_segment_chain(
    paths: Sequence[Path], *, coordinate_id: str, terminal_policy: str
) -> dict[str, Any]:
    """Verify one terminal segment chain and independently finalize economics."""

    segment_paths = [Path(path) for path in paths]
    if not segment_paths or len(segment_paths) > MAX_SEGMENT_PATHS:
        raise DojoEconomicV2AuditorError("segment path denominator is outside its bound")
    if terminal_policy not in {
        MONTH_END_FLAT_SETTLEMENT,
        MONTH_END_MTM_WITH_STATE_HANDOFF,
    }:
        raise DojoEconomicV2AuditorError("terminal policy is unsupported")
    if not coordinate_id or coordinate_id != coordinate_id.strip():
        raise DojoEconomicV2AuditorError("coordinate id must be a trimmed identifier")
    chain = verify_economic_segment_chain(segment_paths, require_terminal=True)
    last = verify_economic_segment(segment_paths[-1])
    session = PortfolioReplaySession.restore_checkpoint(
        policy=last["portfolio_policy"],
        checkpoint=last["terminal_checkpoint"],
    )
    result = validate_portfolio_replay_result(
        session.finalize(terminal_policy=terminal_policy)
    )
    if (
        result["processed_coordinate_count"]
        != chain["completed_coordinate_count"]
        or result["policy_sha256"] != chain["policy_sha256"]
        or last["terminal_source_batch_chain_sha256"]
        != chain["terminal_source_batch_chain_sha256"]
    ):
        raise DojoEconomicV2AuditorError(
            "terminal reducer result differs from the verified segment chain"
        )
    file_chain = GENESIS_SHA256
    total_bytes = 0
    for index, path in enumerate(segment_paths):
        verified_segment = verify_economic_segment(path)
        if any(
            batch["quote"]["coordinate_id"] != coordinate_id
            for batch in verified_segment["batches"]
        ):
            raise DojoEconomicV2AuditorError(
                "segment quote coordinate differs from the fixed account coordinate"
            )
        file_sha, byte_count = _file_digest(path)
        expected_file_sha = hashlib.sha256(
            json.dumps(
                verified_segment,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        ).hexdigest()
        if file_sha != expected_file_sha:
            raise DojoEconomicV2AuditorError(
                "segment file bytes differ from the independently verified value"
            )
        total_bytes += byte_count
        file_chain = canonical_portfolio_sha256(
            {
                "previous_segment_file_chain_sha256": file_chain,
                "segment_index": index,
                "segment_filename": path.name,
                "segment_file_sha256": file_sha,
                "segment_file_bytes": byte_count,
            }
        )
    body: dict[str, Any] = {
        "contract": V2_REEXECUTION_ATTESTATION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "status": "VERIFIED_COMPLETE",
        "transcript_id": chain["transcript_id"],
        "coordinate_id": coordinate_id,
        "job_sha256": chain["job_sha256"],
        "policy_sha256": chain["policy_sha256"],
        "segment_count": chain["segment_count"],
        "segment_chain_tip_sha256": chain["segment_chain_tip_sha256"],
        "segment_file_chain_sha256": file_chain,
        "segment_file_bytes": total_bytes,
        "completed_coordinate_count": chain["completed_coordinate_count"],
        "source_batch_chain_sha256": chain[
            "terminal_source_batch_chain_sha256"
        ],
        "terminal_policy": terminal_policy,
        "portfolio_result_sha256": result["result_sha256"],
        "portfolio_carry_state_sha256": result["carry_state_sha256"],
        "transcript_integrity_passed": True,
        "independent_economic_reexecution_passed": True,
        "partial_economics_reported": False,
        "external_monotonic_anchor_configured": False,
        "fork_absence_proven": False,
        "official_evidence_eligible": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_authority": "NONE",
    }
    body["reexecution_attestation_sha256"] = canonical_portfolio_sha256(body)
    return body


def _canonical_line(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--terminal-policy",
        required=True,
        choices=(
            MONTH_END_FLAT_SETTLEMENT,
            MONTH_END_MTM_WITH_STATE_HANDOFF,
        ),
    )
    parser.add_argument("--coordinate-id", required=True)
    parser.add_argument("segments", nargs="+")
    args = parser.parse_args(argv)
    attestation = reexecute_v2_economic_segment_chain(
        [Path(item) for item in args.segments],
        coordinate_id=args.coordinate_id,
        terminal_policy=args.terminal_policy,
    )
    print(_canonical_line(attestation), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by runner subprocess
    raise SystemExit(main())


__all__ = [
    "DojoEconomicV2AuditorError",
    "V2_REEXECUTION_ATTESTATION_CONTRACT",
    "reexecute_v2_economic_segment_chain",
]
