#!/usr/bin/env python3
"""Paper-only CLI for the future DOJO AI inventory decision ledger.

This command can validate or append a consumable V2 decision.  It has no
broker client and cannot itself consume a decision or mutate a virtual/live
position.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.analysis.market_status import compute_market_status  # noqa: E402
from quant_rabbit.dojo_ai_inventory import (  # noqa: E402
    DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
    DOJO_AI_INVENTORY_DECISION_CONTRACT,
    DOJO_AI_INVENTORY_DECISION_ROLE,
    append_inventory_decision,
    seal_inventory_decision_proposal,
    validate_inventory_decision_ledger,
)


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("decision input must be a JSON object")
    return value


def _market_preflight(
    now_utc: datetime | None = None,
) -> dict[str, object]:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    status = compute_market_status(now)
    return {
        "contract": "QR_DOJO_AI_INVENTORY_PREFLIGHT_V1",
        "decision_contract": DOJO_AI_INVENTORY_DECISION_CONTRACT,
        "consumer_contract": DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
        "decision_role": DOJO_AI_INVENTORY_DECISION_ROLE,
        "as_of_utc": now.isoformat(),
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "external_broker_mutation_allowed": False,
        "virtual_broker_mutation_allowed": status.is_fx_open,
        "fx_market_open": status.is_fx_open,
        "ai_assessment_allowed": status.is_fx_open,
        "ai_inventory_decision_allowed": status.is_fx_open,
        "virtual_action_allowed": False,
        "status": (
            "READY_FOR_DECISION_WRITER"
            if status.is_fx_open
            else "MARKET_CLOSED_AI_INVENTORY_PAUSED"
        ),
        "closed_reason": status.closed_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("health")
    validate = commands.add_parser("validate-ledger")
    validate.add_argument("ledger", type=Path)
    seal = commands.add_parser("seal")
    seal.add_argument("input", type=Path)
    append = commands.add_parser("append")
    append.add_argument("ledger", type=Path)
    append.add_argument("input", type=Path)
    args = parser.parse_args()

    if args.command == "health":
        result = _market_preflight()
    elif args.command == "validate-ledger":
        result = validate_inventory_decision_ledger(args.ledger)
    elif args.command == "seal":
        result = seal_inventory_decision_proposal(_read_object(args.input))
    else:
        preflight = _market_preflight()
        if not preflight["fx_market_open"]:
            raise RuntimeError("MARKET_CLOSED_AI_INVENTORY_PAUSED")
        append_result = append_inventory_decision(
            args.ledger,
            _read_object(args.input),
        )
        result = {
            "appended": append_result.appended,
            "decision_sha256": append_result.record["decision_sha256"],
            "decision_identity_sha256": append_result.record[
                "decision_identity_sha256"
            ],
            "sequence": append_result.record["sequence"],
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "virtual_broker_mutation_allowed": True,
            "external_broker_mutation_allowed": False,
            "consumer_contract": DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
            "decision_role": DOJO_AI_INVENTORY_DECISION_ROLE,
            "virtual_action_allowed": True,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
