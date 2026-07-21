"""Compact transcript adapter for the long-horizon economic runner.

The adapter exposes the existing runner recorder lifecycle while buffering a
bounded number of complete coordinate transactions into immutable V3 segment
documents.  Empty worker proposals are implicit, so the runner no longer emits
four fsynced JSONL records for every quote/configuration transition.

This is a research evidence writer only.  It has no broker, worker, model,
allocation, promotion, or live authority.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_economic_transcript_v2 import (
    MAX_SEGMENT_COORDINATES,
    DojoEconomicSegmentError,
    EconomicSegmentWriter,
)
from quant_rabbit.dojo_portfolio_replay_reducer import (
    PortfolioReplaySession,
    canonical_portfolio_sha256,
    validate_portfolio_replay_result,
    verify_portfolio_policy,
    verify_portfolio_replay_checkpoint,
)


LONG_HORIZON_V2_MANIFEST_CONTRACT: Final = (
    "QR_DOJO_LONG_HORIZON_ECONOMIC_TRANSCRIPT_V3_MANIFEST_V1"
)
SCHEMA_VERSION: Final = 1
DEFAULT_SEGMENT_COORDINATES: Final = 256
GENESIS_SHA256: Final = "0" * 64
MAX_MANIFEST_BYTES: Final = 8 * 1024 * 1024


class DojoLongHorizonV2TranscriptError(ValueError):
    """The compact transcript lifecycle is incomplete or inconsistent."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoLongHorizonV2TranscriptError(
            "compact transcript value is not strict JSON"
        ) from exc


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(value) + b"\n"
    if len(raw) > MAX_MANIFEST_BYTES:
        raise DojoLongHorizonV2TranscriptError(
            "compact transcript manifest exceeds its byte bound"
        )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(raw)
        written = 0
        while written < len(raw):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short compact transcript manifest publication")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class CompactEconomicTranscriptV2Recorder:
    """Adapt one reducer session to bounded immutable V2 segment files."""

    def __init__(
        self,
        *,
        evidence_root: Path,
        transcript_id: str,
        job_sha256: str,
        source_slice_receipt_sha256: str,
        portfolio_policy: Mapping[str, Any],
        expected_quote_batch_count: int,
        session: PortfolioReplaySession,
        segment_coordinate_limit: int = DEFAULT_SEGMENT_COORDINATES,
    ) -> None:
        if (
            isinstance(expected_quote_batch_count, bool)
            or not isinstance(expected_quote_batch_count, int)
            or expected_quote_batch_count < 1
        ):
            raise DojoLongHorizonV2TranscriptError(
                "expected quote-batch denominator must be positive"
            )
        if (
            isinstance(segment_coordinate_limit, bool)
            or not isinstance(segment_coordinate_limit, int)
            or not 1 <= segment_coordinate_limit <= MAX_SEGMENT_COORDINATES
        ):
            raise DojoLongHorizonV2TranscriptError(
                "segment coordinate limit is outside the V2 bound"
            )
        self.evidence_root = Path(evidence_root)
        self.transcript_id = transcript_id
        self.job_sha256 = job_sha256
        self.source_slice_receipt_sha256 = source_slice_receipt_sha256
        self.policy = verify_portfolio_policy(portfolio_policy)
        self.expected_quote_batch_count = expected_quote_batch_count
        self.session = session
        self.segment_coordinate_limit = segment_coordinate_limit
        self._start_checkpoint = verify_portfolio_replay_checkpoint(
            policy=self.policy,
            checkpoint=session.export_checkpoint(),
        )
        self._latest_checkpoint = self._start_checkpoint
        self._prior_segment_sha256 = GENESIS_SHA256
        self._prior_source_batch_chain_sha256 = GENESIS_SHA256
        self._last_completed_source_batch_chain_sha256 = GENESIS_SHA256
        self._segment_index = 0
        self._segment_coordinate_start = 0
        self._completed_coordinate_count = 0
        self._buffer: list[dict[str, Any]] = []
        self._segment_rows: list[dict[str, Any]] = []
        self._pending: dict[str, Any] | None = None
        self._stage = "EXPECT_QUOTE_OR_TERMINAL"
        self._terminal = False
        self._manifest: dict[str, Any] | None = None

    @property
    def segment_paths(self) -> list[Path]:
        return [self.evidence_root / row["segment_filename"] for row in self._segment_rows]

    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            raise DojoLongHorizonV2TranscriptError(
                "compact transcript has no terminal manifest"
            )
        return dict(self._manifest)

    @property
    def manifest_path(self) -> Path:
        return self.evidence_root / f"{self.transcript_id}.economic-v2.manifest.json"

    def _flush(self, *, terminal_segment: bool) -> None:
        if not self._buffer:
            raise DojoLongHorizonV2TranscriptError(
                "cannot publish an empty compact transcript segment"
            )
        source_start = self._buffer[0]["source_offset_start"]
        source_end = self._buffer[-1]["source_offset_end_exclusive"]
        path = self.evidence_root / (
            f"{self.transcript_id}.segment-{self._segment_index:06d}.economic-v2.json"
        )
        try:
            segment = EconomicSegmentWriter(path).publish(
                transcript_id=self.transcript_id,
                job_sha256=self.job_sha256,
                segment_index=self._segment_index,
                prior_segment_sha256=self._prior_segment_sha256,
                portfolio_policy=self.policy,
                source_slice_receipt_sha256=self.source_slice_receipt_sha256,
                source_offset_start=source_start,
                source_offset_end_exclusive=source_end,
                prior_source_batch_chain_sha256=(
                    self._prior_source_batch_chain_sha256
                ),
                expected_job_coordinate_count=self.expected_quote_batch_count,
                segment_coordinate_start=self._segment_coordinate_start,
                start_checkpoint=self._start_checkpoint,
                terminal_checkpoint=self._latest_checkpoint,
                batches=self._buffer,
                terminal_segment=terminal_segment,
            )
        except DojoEconomicSegmentError as exc:
            raise DojoLongHorizonV2TranscriptError(
                "compact economic segment publication failed"
            ) from exc
        self._segment_rows.append(
            {
                "segment_index": self._segment_index,
                "segment_filename": path.name,
                "segment_sha256": segment["segment_sha256"],
                "segment_coordinate_start": self._segment_coordinate_start,
                "segment_coordinate_end_exclusive": self._completed_coordinate_count,
                "segment_file_bytes": path.stat(follow_symlinks=False).st_size,
                "terminal_segment": terminal_segment,
            }
        )
        self._prior_segment_sha256 = segment["segment_sha256"]
        self._prior_source_batch_chain_sha256 = segment[
            "terminal_source_batch_chain_sha256"
        ]
        self._start_checkpoint = self._latest_checkpoint
        self._segment_coordinate_start = self._completed_coordinate_count
        self._segment_index += 1
        self._buffer = []

    def record_quote_batch(
        self,
        *,
        coordinate_id: str,
        epoch: int,
        phase: str,
        intrabar: str,
        quote_watermark: int,
        quotes: Sequence[Mapping[str, Any]],
        quote_batch_sha256_value: str,
        source_batch_chain_sha256: str,
    ) -> dict[str, Any]:
        if self._terminal or self._stage != "EXPECT_QUOTE_OR_TERMINAL":
            raise DojoLongHorizonV2TranscriptError("compact quote batch is out of order")
        if len(self._buffer) >= self.segment_coordinate_limit:
            self._flush(terminal_segment=False)
        ordinal = self._completed_coordinate_count
        quote = {
            "coordinate_id": coordinate_id,
            "epoch": epoch,
            "phase": phase,
            "intrabar": intrabar,
            "quote_watermark": quote_watermark,
            "quotes": [dict(row) for row in quotes],
            "quote_batch_sha256": quote_batch_sha256_value,
            "source_batch_chain_sha256": source_batch_chain_sha256,
        }
        self._pending = {
            "coordinate_ordinal": ordinal,
            # V2 runner integration uses normalized quote-batch ordinals.  The
            # immutable source-slice receipt binds the original file bytes;
            # quotes and their causal order are independently bound by the
            # quote/source chains.
            "source_offset_start": ordinal,
            "source_offset_end_exclusive": ordinal + 1,
            "quote": quote,
        }
        self._stage = "EXPECT_SNAPSHOT"
        return dict(quote)

    def record_post_exit_snapshot(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        if self._stage != "EXPECT_SNAPSHOT" or self._pending is None:
            raise DojoLongHorizonV2TranscriptError(
                "compact post-exit snapshot is out of order"
            )
        self._pending["post_exit_snapshot"] = dict(snapshot)
        self._stage = "EXPECT_PROPOSALS"
        return dict(snapshot)

    def record_worker_proposal_batch(
        self, proposal_batch: Mapping[str, Any]
    ) -> dict[str, Any]:
        if self._stage != "EXPECT_PROPOSALS" or self._pending is None:
            raise DojoLongHorizonV2TranscriptError(
                "compact proposal batch is out of order"
            )
        proposals = proposal_batch.get("proposals")
        if isinstance(proposals, (str, bytes, bytearray)) or not isinstance(
            proposals, Sequence
        ):
            raise DojoLongHorizonV2TranscriptError(
                "compact proposal batch has no bounded proposal sequence"
            )
        non_hold = [
            dict(row)
            for row in proposals
            if row["intent_counts"]["risk_reducing"]
            or row["intent_counts"]["new_risk"]
        ]
        self._pending["non_hold_proposals"] = non_hold
        self._stage = "EXPECT_ALLOCATION"
        return dict(proposal_batch)

    def record_allocation_receipt(self, receipt: Mapping[str, Any]) -> dict[str, Any]:
        if self._stage != "EXPECT_ALLOCATION" or self._pending is None:
            raise DojoLongHorizonV2TranscriptError(
                "compact allocation receipt is out of order"
            )
        self._pending["allocation_receipt"] = dict(receipt)
        unsigned = dict(self._pending)
        self._pending["batch_sha256"] = canonical_portfolio_sha256(unsigned)
        self._buffer.append(self._pending)
        self._completed_coordinate_count += 1
        self._latest_checkpoint = verify_portfolio_replay_checkpoint(
            policy=self.policy,
            checkpoint=self.session.export_checkpoint(),
        )
        self._last_completed_source_batch_chain_sha256 = self._pending["quote"][
            "source_batch_chain_sha256"
        ]
        self._pending = None
        self._stage = "EXPECT_QUOTE_OR_TERMINAL"
        return dict(receipt)

    def _publish_manifest(
        self,
        *,
        terminal_status: str,
        failure_code: str | None,
        failure_stage: str | None,
        failure_evidence_sha256: str | None,
        portfolio_result_sha256: str | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "contract": LONG_HORIZON_V2_MANIFEST_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "transcript_id": self.transcript_id,
            "job_sha256": self.job_sha256,
            "policy_sha256": self.policy["policy_sha256"],
            "source_slice_receipt_sha256": self.source_slice_receipt_sha256,
            "source_range_unit": "NORMALIZED_QUOTE_BATCH_ORDINAL",
            "expected_quote_batch_count": self.expected_quote_batch_count,
            "completed_quote_batch_count": self._completed_coordinate_count,
            "terminal_status": terminal_status,
            "failure_code": failure_code,
            "failure_stage": failure_stage,
            "failure_evidence_sha256": failure_evidence_sha256,
            "portfolio_result_sha256": portfolio_result_sha256,
            "terminal_source_batch_chain_sha256": (
                self._last_completed_source_batch_chain_sha256
            ),
            "segment_count": len(self._segment_rows),
            "segments": self._segment_rows,
            "segments_sha256": canonical_portfolio_sha256(self._segment_rows),
            "partial_economics_reported": False,
            "external_monotonic_anchor_configured": False,
            "fork_absence_proven": False,
            "official_evidence_eligible": False,
            "live_permission": False,
            "broker_mutation_allowed": False,
            "order_authority": "NONE",
        }
        body["manifest_sha256"] = canonical_portfolio_sha256(body)
        _write_exclusive(self.manifest_path, body)
        self._manifest = body
        return dict(body)

    def seal_success(
        self,
        *,
        terminal_policy: str,
        portfolio_result: Mapping[str, Any],
        source_batch_chain_sha256: str,
    ) -> dict[str, Any]:
        if (
            self._terminal
            or self._stage != "EXPECT_QUOTE_OR_TERMINAL"
            or self._completed_coordinate_count != self.expected_quote_batch_count
            or not self._buffer
        ):
            raise DojoLongHorizonV2TranscriptError(
                "compact terminal success denominator is incomplete"
            )
        result = validate_portfolio_replay_result(portfolio_result)
        if (
            result["terminal_policy"] != terminal_policy
            or result["policy_sha256"] != self.policy["policy_sha256"]
            or result["processed_coordinate_count"]
            != self._completed_coordinate_count
            or source_batch_chain_sha256
            != self._last_completed_source_batch_chain_sha256
        ):
            raise DojoLongHorizonV2TranscriptError(
                "compact terminal result binding drifted"
            )
        self._flush(terminal_segment=True)
        self._terminal = True
        return self._publish_manifest(
            terminal_status="COMPLETE",
            failure_code=None,
            failure_stage=None,
            failure_evidence_sha256=None,
            portfolio_result_sha256=result["result_sha256"],
        )

    def seal_failure(
        self,
        *,
        failure_code: str,
        failure_stage: str,
        failure_evidence_sha256: str,
    ) -> dict[str, Any]:
        if self._terminal:
            raise DojoLongHorizonV2TranscriptError(
                "compact transcript is already terminal"
            )
        self._pending = None
        self._stage = "EXPECT_QUOTE_OR_TERMINAL"
        if self._buffer:
            self._flush(terminal_segment=False)
        self._terminal = True
        return self._publish_manifest(
            terminal_status="FAILED",
            failure_code=failure_code,
            failure_stage=failure_stage,
            failure_evidence_sha256=failure_evidence_sha256,
            portfolio_result_sha256=None,
        )

    def close(self) -> None:
        self._pending = None


__all__ = [
    "CompactEconomicTranscriptV2Recorder",
    "DEFAULT_SEGMENT_COORDINATES",
    "DojoLongHorizonV2TranscriptError",
    "LONG_HORIZON_V2_MANIFEST_CONTRACT",
]
