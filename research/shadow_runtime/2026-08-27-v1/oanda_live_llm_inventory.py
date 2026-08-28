"""One-shot actual-LLM inventory receipt over a hash-fixed shadow snapshot."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from shadow_runtime import HashLedger, canonical_bytes, canonical_hash, parse_utc, utc_text

MODEL = "gpt-5.6-sol"
MODEL_VERSION = "gpt-5.6-sol"
REASONING = "high"
SOURCE_COMMIT = "9b2be18dd2440ffcd13d25af1421684328f52ee0"
COMPLETED_M5_HEAD = "1e09c00a91157dff885b80332abcfed357355b5c33ff901b3737df0acd78e993"
CANARY_HEAD = "4fcd63cf93129d28d16cee726c54ed29eb3a4ff7a28886531386c698a3d1f406"
INVENTORY_SNAPSHOT_HASH = "9005ea08e61cd7d79aaaa5fe314905999e5e10d8aa349f8d5e6186283ae449c1"
OUTPUT_FIELDS = {"action", "currency_cap", "mode", "valid_until", "confidence", "reason"}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["ADD", "FREEZE", "UNWIND", "RESET"]},
        "currency_cap": {"type": "integer", "minimum": 0, "maximum": 1000000000},
        "mode": {"type": "string", "enum": ["SHADOW_ONLY"]},
        "valid_until": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "maxLength": 240},
    },
    "required": sorted(OUTPUT_FIELDS),
    "additionalProperties": False,
}


def build_input(run_root: Path, git_head: str) -> dict[str, Any]:
    if git_head != SOURCE_COMMIT:
        raise RuntimeError("SOURCE_COMMIT_MISMATCH")
    completed = HashLedger(run_root / "ledgers" / "completed_m5.jsonl")
    canary = HashLedger(run_root / "ledgers" / "plumbing_canary.jsonl")
    if len(completed.rows) != 2 or completed.last_hash != COMPLETED_M5_HEAD:
        raise RuntimeError("COMPLETED_M5_SNAPSHOT_MISMATCH")
    if len(canary.rows) != 8 or canary.last_hash != CANARY_HEAD:
        raise RuntimeError("CANARY_SNAPSHOT_MISMATCH")
    inventory_rows = [row for row in canary.rows if row["payload"].get("kind") == "inventory_snapshot"]
    if len(inventory_rows) != 1 or inventory_rows[0]["record_hash"] != INVENTORY_SNAPSHOT_HASH:
        raise RuntimeError("INVENTORY_SNAPSHOT_MISMATCH")
    inventory = inventory_rows[0]["payload"]
    return {
        "schema_version": 1,
        "source_commit": SOURCE_COMMIT,
        "strategy_status": "RESEARCH_NOT_ADMITTED",
        "shadow_status": "OBSERVATION_AUTHORIZED",
        "live_order_authority": False,
        "external_orders": 0,
        "r5_candidate_scope": "ACCOUNTING_ONLY_NOT_CAUSAL_SIGNAL_ADMISSION",
        "profit_unproven": True,
        "completed_m5_ledger": {"rows": 2, "head": completed.last_hash},
        "inventory_snapshot": {
            "record_hash": inventory_rows[0]["record_hash"],
            "open_inventory_count": int(inventory["open_inventory_count"]),
            "arm": "PLUMBING_CANARY_NON_EVIDENCE",
            "evidence_eligible": False,
            "profit_evidence": False,
            "individual_order_attributes_included": False,
            "canary_pnl_included": False,
        },
        "frozen_inventory_policy": {
            "policy_id": "FROZEN_INVENTORY_POLICY_V2",
            "max_currency_notional_jpy_micros": 1000000000,
            "max_gross_notional_jpy_micros": 1000000000,
            "max_open_positions_per_arm": 1,
            "same_pair_collision": "REJECT_NEW",
            "terminal_liquidation": True,
        },
        "allowed_output": {
            "actions": ["ADD", "FREEZE", "UNWIND", "RESET"],
            "currency_cap_range_jpy_micros": [0, 1000000000],
            "mode": "SHADOW_ONLY",
        },
    }


def build_prompt(snapshot: dict[str, Any], request_time: datetime) -> str:
    valid_until_limit = request_time + timedelta(hours=2)
    return (
        "You are the actual LLM inventory-policy worker for a zero-authority FX shadow. "
        "Return exactly one JSON object matching the supplied schema. You may choose only "
        "ADD, FREEZE, UNWIND, or RESET; set currency_cap in JPY microunits within the frozen "
        "policy; mode must be SHADOW_ONLY; valid_until must be RFC3339 UTC after request_time "
        "and no later than valid_until_limit; confidence is 0..1; reason is at most 240 characters. "
        "You control only aggregate virtual inventory. You must not create or alter any individual "
        "order, direction, fill, TP, SL, leverage, execution cost, or hard guard. External orders "
        "remain zero. The snapshot is a non-evidence plumbing canary; do not treat its PnL or its "
        "existence as profit or R5 adoption evidence.\n"
        f"request_time_utc={utc_text(request_time)}\n"
        f"valid_until_limit_utc={utc_text(valid_until_limit)}\n"
        f"snapshot={canonical_bytes(snapshot).decode()}"
    )


def validate_output(output: dict[str, Any], request_time: datetime) -> None:
    if set(output) != OUTPUT_FIELDS:
        raise RuntimeError("LLM_OUTPUT_FIELDS_INVALID")
    if output["action"] not in {"ADD", "FREEZE", "UNWIND", "RESET"}:
        raise RuntimeError("LLM_ACTION_INVALID")
    if type(output["currency_cap"]) is not int or not 0 <= output["currency_cap"] <= 1000000000:
        raise RuntimeError("LLM_CAP_INVALID")
    if output["mode"] != "SHADOW_ONLY":
        raise RuntimeError("LLM_MODE_INVALID")
    valid_until = parse_utc(output["valid_until"])
    if not request_time < valid_until <= request_time + timedelta(hours=2):
        raise RuntimeError("LLM_VALID_UNTIL_INVALID")
    if not isinstance(output["confidence"], (int, float)) or not 0 <= output["confidence"] <= 1:
        raise RuntimeError("LLM_CONFIDENCE_INVALID")
    if not isinstance(output["reason"], str) or not 1 <= len(output["reason"]) <= 240:
        raise RuntimeError("LLM_REASON_INVALID")


def run_once(
    run_root: Path,
    git_head: str,
    output: dict[str, Any],
    decision_time: datetime,
    arrival_time: datetime,
) -> dict[str, Any]:
    receipt_ledger = HashLedger(run_root / "ledgers" / "llm_inventory_receipts.jsonl")
    if receipt_ledger.rows:
        raise RuntimeError("LLM_CALL_ALREADY_RECORDED")
    snapshot = build_input(run_root, git_head)
    prompt = build_prompt(snapshot, decision_time)
    validate_output(output, decision_time)
    if arrival_time < decision_time:
        raise RuntimeError("LLM_ARRIVAL_TIME_INVALID")
    payload = {
        "schema_version": 1,
        "kind": "ACTUAL_LLM_INVENTORY_RECEIPT",
        "model": MODEL,
        "model_version": MODEL_VERSION,
        "reasoning": REASONING,
        "prompt_full": prompt,
        "prompt_sha256": canonical_hash(prompt),
        "input_snapshot": snapshot,
        "input_sha256": canonical_hash(snapshot),
        "output": output,
        "output_sha256": canonical_hash(output),
        "decision_timestamp_utc": utc_text(decision_time),
        "arrival_timestamp_utc": utc_text(arrival_time),
        "allowed_output_fields": sorted(OUTPUT_FIELDS),
        "individual_order_control": False,
        "hard_guard_mutation": False,
        "canary_profit_evidence": False,
        "r5_result_included": False,
        "llm_calls": 1,
        "external_order_attempts": 0,
        "external_orders": 0,
    }
    planned: list[dict[str, Any]] = []
    row = receipt_ledger.plan(payload, f"actual-llm::{payload['input_sha256']}", planned)
    receipt_ledger.append_rows(planned)
    return {"receipt_record_hash": row["record_hash"], **payload}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--git-head", required=True)
    parser.add_argument("--decision-json", required=True)
    parser.add_argument("--decision-time", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_once(
            args.run_root,
            args.git_head,
            json.loads(args.decision_json),
            parse_utc(args.decision_time),
            datetime.now(timezone.utc),
        )
        print(json.dumps({
            "model": result["model"],
            "model_version": result["model_version"],
            "reasoning": result["reasoning"],
            "action": result["output"]["action"],
            "currency_cap": result["output"]["currency_cap"],
            "mode": result["output"]["mode"],
            "valid_until": result["output"]["valid_until"],
            "confidence": result["output"]["confidence"],
            "prompt_sha256": result["prompt_sha256"],
            "input_sha256": result["input_sha256"],
            "output_sha256": result["output_sha256"],
            "receipt_record_hash": result["receipt_record_hash"],
            "llm_calls": result["llm_calls"],
            "external_orders": result["external_orders"],
        }, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
