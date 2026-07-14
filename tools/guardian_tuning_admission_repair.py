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

from tools.guardian_wake_dispatcher import repair_tuning_queue_admissions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and remove non-tuning Guardian events that were admitted "
            "to the hourly tuning queue by an older dispatcher."
        )
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=ROOT / "data" / "guardian_tuning_work_order.json",
    )
    args = parser.parse_args()
    result = repair_tuning_queue_admissions(
        path=args.queue.resolve(),
        now=datetime.now(timezone.utc),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") in {
        "ADMISSION_REPAIR_NOT_REQUIRED",
        "ADMISSION_REPAIR_WRITTEN",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
