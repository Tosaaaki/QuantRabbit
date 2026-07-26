#!/usr/bin/env python3
"""Run one future, isolated, PAPER_ELIGIBLE paper-AI inventory room."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from quant_rabbit.dojo_ai_inventory_session import (  # noqa: E402
    AIInventorySessionError,
    load_ai_inventory_session_config,
    run_registered_ai_inventory_session,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Canonical direct child of config/paper_ai_inventory/",
    )
    args = parser.parse_args(argv)
    try:
        config = load_ai_inventory_session_config(REPOSITORY_ROOT, args.config)
        result = run_registered_ai_inventory_session(
            config,
            screen_identity=os.environ.get("STY"),
            process_argv=tuple(sys.argv),
        )
    except AIInventorySessionError as exc:
        print(
            json.dumps(
                {
                    "contract": "QR_DOJO_AI_INVENTORY_SESSION_RUNNER_RESULT_V1",
                    "status": "FAIL_CLOSED",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "paper_only": True,
                    "order_authority": "NONE",
                    "live_permission": False,
                    "external_broker_mutation_allowed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "contract": "QR_DOJO_AI_INVENTORY_SESSION_RUNNER_RESULT_V1",
                "status": result.status,
                "room_root": str(result.room_root),
                "lifecycle_tip_sha256": result.lifecycle_tip_sha256,
                "positions_count": result.positions_count,
                "orders_count": result.orders_count,
                "pending_evaluations": result.pending_evaluations,
                "paper_only": True,
                "order_authority": "NONE",
                "live_permission": False,
                "external_broker_mutation_allowed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
