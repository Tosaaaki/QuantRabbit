#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.guardian_wake_dispatcher import enrich_tuning_work_order_review  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Safely bind a pending guardian observation to one TEST_REQUIRED review."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=ROOT / "data" / "guardian_tuning_work_order.json",
    )
    parser.add_argument("--work-order-id", required=True)
    parser.add_argument("--expected-observation-id", required=True)
    parser.add_argument("--review-json", type=Path, required=True)
    parser.add_argument("--reviewed-by", required=True)
    args = parser.parse_args(argv)
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
        reviewed_by=args.reviewed_by,
        now=datetime.now(timezone.utc),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") in {
        "WORK_ORDER_REVIEW_ENRICHED",
        "WORK_ORDER_REVIEW_ALREADY_BOUND",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
