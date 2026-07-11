#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.guardian_wake_dispatcher import (  # noqa: E402
    MAX_PENDING_TUNING_WORK_ORDERS,
    MAX_TUNING_QUEUE_BYTES,
    TUNING_REVIEW_BATCH_ITEM_FIELDS,
    _apply_tuning_work_order_review,
    _load_tuning_work_order,
    _normalized_tuning_work_order_queue,
    _tuning_queue_json_shape_error,
    enrich_tuning_work_order_review,
    enrich_tuning_work_order_reviews_batch,
)


_BATCH_ITEM_FIELDS = TUNING_REVIEW_BATCH_ITEM_FIELDS
_SUCCESS_STATUSES = {
    "WORK_ORDER_REVIEW_ENRICHED",
    "WORK_ORDER_REVIEW_ALREADY_BOUND",
}


def _batch_failure(
    *,
    index: int | None,
    code: str,
    work_order_id: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code}
    if index is not None:
        result["index"] = index
    if work_order_id:
        result["work_order_id"] = work_order_id
    result.update(details)
    return result


def _read_batch_manifest_json(path: Path) -> tuple[Any | None, dict[str, Any] | None]:
    """Read a manifest under the queue's raw-size and JSON-shape bounds."""

    try:
        with path.open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            if size < 0 or size > MAX_TUNING_QUEUE_BYTES:
                return None, {
                    "status": "MANIFEST_JSON_TOO_LARGE",
                    "max_bytes": MAX_TUNING_QUEUE_BYTES,
                    "observed_bytes": size,
                }
            raw = handle.read(MAX_TUNING_QUEUE_BYTES + 1)
    except OSError as exc:
        return None, {
            "status": "MANIFEST_JSON_READ_FAILED",
            "error": f"{type(exc).__name__}: {exc}",
        }
    if len(raw) > MAX_TUNING_QUEUE_BYTES:
        return None, {
            "status": "MANIFEST_JSON_TOO_LARGE",
            "max_bytes": MAX_TUNING_QUEUE_BYTES,
            "observed_bytes": len(raw),
        }
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        return None, {
            "status": "MANIFEST_JSON_READ_FAILED",
            "error": f"{type(exc).__name__}: invalid manifest JSON",
        }
    shape_error = _tuning_queue_json_shape_error(manifest)
    if shape_error is not None:
        return None, {
            "status": "MANIFEST_JSON_SHAPE_INVALID",
            "error": shape_error,
        }
    return manifest, None


def _prevalidate_batch_manifest(
    *,
    path: Path,
    manifest: Any,
) -> dict[str, Any]:
    """Validate every batch item before any review writer is called."""

    failures: list[dict[str, Any]] = []
    if not isinstance(manifest, dict):
        failures.append(
            _batch_failure(index=None, code="MANIFEST_ROOT_NOT_OBJECT")
        )
        return {
            "status": "BATCH_MANIFEST_VALIDATION_FAILED",
            "requested_count": 0,
            "validated_count": 0,
            "written_count": 0,
            "failures": failures,
        }
    unexpected_root = sorted(set(manifest) - {"reviews"})
    if unexpected_root:
        failures.append(
            _batch_failure(
                index=None,
                code="MANIFEST_ROOT_FIELDS_INVALID",
                unexpected_fields=unexpected_root,
            )
        )
    reviews = manifest.get("reviews")
    if not isinstance(reviews, list) or not reviews:
        failures.append(
            _batch_failure(index=None, code="MANIFEST_REVIEWS_REQUIRED")
        )
        reviews = []
    elif len(reviews) > MAX_PENDING_TUNING_WORK_ORDERS:
        failures.append(
            _batch_failure(
                index=None,
                code="MANIFEST_REVIEW_COUNT_EXCEEDED",
                review_count=len(reviews),
                max_review_count=MAX_PENDING_TUNING_WORK_ORDERS,
            )
        )

    seen_work_order_ids: set[str] = set()
    seen_observation_ids: set[str] = set()
    structured: list[dict[str, Any]] = []
    for index, item in enumerate(reviews):
        if not isinstance(item, dict):
            failures.append(
                _batch_failure(index=index, code="MANIFEST_ITEM_NOT_OBJECT")
            )
            continue
        unexpected = sorted(set(item) - _BATCH_ITEM_FIELDS)
        missing = sorted(_BATCH_ITEM_FIELDS - set(item))
        work_order_id = str(item.get("work_order_id") or "").strip()
        observation_id = str(item.get("expected_observation_id") or "").strip()
        if unexpected or missing:
            failures.append(
                _batch_failure(
                    index=index,
                    code="MANIFEST_ITEM_FIELDS_INVALID",
                    work_order_id=work_order_id or None,
                    unexpected_fields=unexpected,
                    missing_fields=missing,
                )
            )
            continue
        if not work_order_id or not observation_id:
            failures.append(
                _batch_failure(
                    index=index,
                    code="MANIFEST_ITEM_ID_REQUIRED",
                    work_order_id=work_order_id or None,
                )
            )
            continue
        if work_order_id in seen_work_order_ids:
            failures.append(
                _batch_failure(
                    index=index,
                    code="DUPLICATE_WORK_ORDER_ID",
                    work_order_id=work_order_id,
                )
            )
            continue
        if observation_id in seen_observation_ids:
            failures.append(
                _batch_failure(
                    index=index,
                    code="DUPLICATE_OBSERVATION_ID",
                    work_order_id=work_order_id,
                    expected_observation_id=observation_id,
                )
            )
            continue
        seen_work_order_ids.add(work_order_id)
        seen_observation_ids.add(observation_id)
        structured.append(
            {
                "index": index,
                "work_order_id": work_order_id,
                "expected_observation_id": observation_id,
                "review": item.get("review"),
            }
        )

    if failures:
        return {
            "status": "BATCH_MANIFEST_VALIDATION_FAILED",
            "requested_count": len(reviews),
            "validated_count": 0,
            "written_count": 0,
            "failures": failures,
        }

    loaded = _load_tuning_work_order(path)
    if loaded.get("_read_error"):
        return {
            "status": "BATCH_MANIFEST_VALIDATION_FAILED",
            "requested_count": len(reviews),
            "validated_count": 0,
            "written_count": 0,
            "failures": [
                _batch_failure(
                    index=None,
                    code="WORK_ORDER_READ_FAILED",
                    error=loaded.get("_read_error"),
                )
            ],
        }
    try:
        pending, _ = _normalized_tuning_work_order_queue(loaded)
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        return {
            "status": "BATCH_MANIFEST_VALIDATION_FAILED",
            "requested_count": len(reviews),
            "validated_count": 0,
            "written_count": 0,
            "failures": [
                _batch_failure(
                    index=None,
                    code="WORK_ORDER_QUEUE_INVALID",
                    error=f"{type(exc).__name__}: {exc}",
                )
            ],
        }

    validated: list[dict[str, Any]] = []
    updated_pending = pending
    preview_now = datetime.now(timezone.utc)
    for item in structured:
        result, candidate_pending, candidate_primary = (
            _apply_tuning_work_order_review(
                path=path,
                pending_entries=updated_pending,
                work_order_id=item["work_order_id"],
                expected_observation_id=item["expected_observation_id"],
                review=item["review"],
                reviewed_by="manifest-prevalidation",
                now=preview_now,
            )
        )
        if result.get("status") not in _SUCCESS_STATUSES:
            failures.append(
                _batch_failure(
                    index=item["index"],
                    code=str(result.get("status") or "BATCH_ITEM_INVALID"),
                    work_order_id=item["work_order_id"],
                    result=result,
                )
            )
            continue
        if result.get("status") == "WORK_ORDER_REVIEW_ENRICHED":
            updated_pending = candidate_pending
        normalized_review = (
            candidate_primary.get("bot_tuning_review")
            if isinstance(candidate_primary, dict)
            and isinstance(candidate_primary.get("bot_tuning_review"), dict)
            else item["review"]
        )
        validated.append(
            {
                "work_order_id": item["work_order_id"],
                "expected_observation_id": item["expected_observation_id"],
                "review": normalized_review,
            }
        )

    if failures:
        return {
            "status": "BATCH_MANIFEST_VALIDATION_FAILED",
            "requested_count": len(reviews),
            "validated_count": len(validated),
            "written_count": 0,
            "failures": failures,
        }
    return {
        "status": "VALID",
        "requested_count": len(reviews),
        "validated_count": len(validated),
        "written_count": 0,
        "reviews": validated,
        "failures": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Safely bind a pending guardian observation to one TEST_REQUIRED or "
            "NO_CHANGE_INSUFFICIENT_EVIDENCE review."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=ROOT / "data" / "guardian_tuning_work_order.json",
    )
    parser.add_argument("--work-order-id")
    parser.add_argument("--expected-observation-id")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--review-json", type=Path)
    input_group.add_argument("--manifest-json", type=Path)
    parser.add_argument("--reviewed-by", required=True)
    args = parser.parse_args(argv)
    reviewed_by = str(args.reviewed_by or "").strip()
    if not reviewed_by:
        print(json.dumps({"status": "REVIEWED_BY_REQUIRED"}))
        return 1

    if args.manifest_json is not None:
        if args.work_order_id is not None or args.expected_observation_id is not None:
            print(
                json.dumps(
                    {
                        "status": "BATCH_ARGUMENT_CONFLICT",
                        "error": (
                            "--work-order-id and --expected-observation-id are single-review "
                            "arguments"
                        ),
                    }
                )
            )
            return 1
        manifest, manifest_error = _read_batch_manifest_json(args.manifest_json)
        if manifest_error is not None:
            print(json.dumps(manifest_error, ensure_ascii=False, sort_keys=True))
            return 1
        prevalidation = _prevalidate_batch_manifest(path=args.path, manifest=manifest)
        if prevalidation.get("status") != "VALID":
            print(json.dumps(prevalidation, ensure_ascii=False, sort_keys=True))
            return 1
        result = enrich_tuning_work_order_reviews_batch(
            path=args.path,
            reviews=prevalidation["reviews"],
            reviewed_by=reviewed_by,
            now=datetime.now(timezone.utc),
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result.get("status") == "BATCH_REVIEW_ENRICHED" else 1

    if not args.work_order_id or not args.expected_observation_id:
        print(
            json.dumps(
                {
                    "status": "SINGLE_REVIEW_ARGUMENTS_REQUIRED",
                    "error": (
                        "--work-order-id and --expected-observation-id are required "
                        "with --review-json"
                    ),
                }
            )
        )
        return 1
    try:
        review = json.loads(args.review_json.read_text())
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "REVIEW_JSON_READ_FAILED", "error": str(exc)}))
        return 1
    result = enrich_tuning_work_order_review(
        path=args.path,
        work_order_id=args.work_order_id,
        expected_observation_id=args.expected_observation_id,
        review=review,
        reviewed_by=reviewed_by,
        now=datetime.now(timezone.utc),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") in _SUCCESS_STATUSES else 1


if __name__ == "__main__":
    raise SystemExit(main())
