#!/usr/bin/env python3
"""Build one canonical forward-only guardian tuning cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from quant_rabbit.guardian_tuning_cohort import (  # noqa: E402
    build_canonical_forward_cohort,
    validate_canonical_forward_cohort,
)
from tools import guardian_wake_dispatcher as dispatcher  # noqa: E402


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_immutable(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("content-addressed artifact collision")
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def _work_order(
    path: Path,
    *,
    work_order_id: str,
    expected_observation_id: str,
) -> dict[str, Any]:
    loaded = dispatcher._load_tuning_work_order(path)
    if loaded.get("_read_error"):
        raise ValueError(str(loaded["_read_error"]))
    pending, _ = dispatcher._normalized_tuning_work_order_queue(loaded)
    matches = [
        item
        for item in pending
        if str(item.get("work_order_id") or "") == work_order_id
    ]
    if len(matches) != 1:
        raise ValueError("exactly one pending work order is required")
    work_order = matches[0]
    latest = str(
        work_order.get("latest_observation_id")
        or work_order.get("observation_id")
        or work_order.get("event_fingerprint")
        or ""
    )
    if latest != expected_observation_id:
        raise ValueError("expected observation is stale")
    validation = work_order.get("bot_tuning_review_validation")
    review = work_order.get("bot_tuning_review")
    if (
        not isinstance(validation, dict)
        or str(validation.get("status") or "").upper() != "VALID"
        or not isinstance(review, dict)
        or str(review.get("review_status") or "").upper() != "TEST_REQUIRED"
        or str(work_order.get("latest_reviewed_observation_id") or "") != latest
    ):
        raise ValueError("current observation needs one valid TEST_REQUIRED review")
    if not str(work_order.get("structured_review_completed_at_utc") or ""):
        raise ValueError("review completion timestamp is missing")
    return work_order


def _lane_id(work_order: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit.strip()
    event = (
        work_order.get("selected_event")
        if isinstance(work_order.get("selected_event"), dict)
        else {}
    )
    details = event.get("details") if isinstance(event.get("details"), dict) else {}
    lane = str(details.get("lane_id") or event.get("lane_id") or "").strip()
    if not lane:
        raise ValueError("exact lane-id is required when the guardian event has no lane_id")
    return lane


def build_cohort(
    *,
    queue_path: Path,
    work_order_id: str,
    expected_observation_id: str,
    ledger_path: Path,
    entry_thesis_path: Path,
    forecast_history_path: Path,
    lane_id: str | None = None,
) -> dict[str, Any]:
    work_order = _work_order(
        queue_path,
        work_order_id=work_order_id,
        expected_observation_id=expected_observation_id,
    )
    review = work_order["bot_tuning_review"]
    return build_canonical_forward_cohort(
        ledger_path=ledger_path,
        entry_thesis_path=entry_thesis_path,
        forecast_history_path=forecast_history_path,
        review=review,
        review_completed_at_utc=work_order["structured_review_completed_at_utc"],
        lane_id=_lane_id(work_order, lane_id),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the first 20 post-review entries for one exact lane from canonical "
            "sources and require all 20 resolutions."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=ROOT / "data" / "guardian_tuning_work_order.json",
    )
    parser.add_argument("--work-order-id", required=True)
    parser.add_argument("--expected-observation-id", required=True)
    parser.add_argument("--lane-id")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "data" / "execution_ledger.db",
    )
    parser.add_argument(
        "--entry-thesis",
        type=Path,
        default=ROOT / "data" / "entry_thesis_ledger.jsonl",
    )
    parser.add_argument(
        "--forecast-history",
        type=Path,
        default=ROOT / "data" / "forecast_history.jsonl",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        payload = build_cohort(
            queue_path=args.path,
            work_order_id=args.work_order_id,
            expected_observation_id=args.expected_observation_id,
            ledger_path=args.ledger,
            entry_thesis_path=args.entry_thesis,
            forecast_history_path=args.forecast_history,
            lane_id=args.lane_id,
        )
        raw = _json_bytes(payload)
        digest = hashlib.sha256(raw).hexdigest()
        expected = ROOT / "data" / "guardian_tuning_cohorts" / f"{digest}.json"
        output = args.output or expected
        if output.resolve() != expected.resolve():
            raise ValueError(f"output must be content-addressed path {expected}")
        _write_immutable(output, raw)
        result = {
            "status": "GUARDIAN_TUNING_FORWARD_COHORT_WRITTEN",
            "path": str(output),
            "sha256": digest,
            "sample_count": len(payload["samples"]),
            "validation_mode": payload["validation_contract"]["mode"],
        }
        code = 0
    except (
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
        sqlite3.Error,
        json.JSONDecodeError,
    ) as exc:
        result = {"status": "GUARDIAN_TUNING_COHORT_FAILED", "error": str(exc)}
        code = 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
