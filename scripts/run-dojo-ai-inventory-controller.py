#!/usr/bin/env python3
"""Run one dormant paper-AI inventory controller cycle.

No broker is started here and no key is accepted on argv. A future isolated
room must already own the separate broker service. The runner HMAC key is read
from the fixed environment variable only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from quant_rabbit.dojo_ai_inventory_broker_service import (
    DojoAIInventoryRunnerClient,
    derive_broker_socket_path,
)
from quant_rabbit.dojo_ai_inventory_controller import (
    controller_config_from_mapping,
    run_ai_inventory_cycle,
)
from quant_rabbit.dojo_replay_lifecycle import canonical_paper_ai_rooms_root


RUNNER_HMAC_ENV = "QR_DOJO_AI_INVENTORY_RUNNER_HMAC_KEY_HEX"
BROKER_LEDGER_NAME = "broker_ledger.jsonl"
MAX_INPUT_BYTES = 2 * 1024 * 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one paper-only AI inventory controller cycle."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--evidence-request", required=True, type=Path)
    args = parser.parse_args(argv)

    repository_root = Path(__file__).resolve(strict=True).parents[1]
    config_value = _read_json(args.config, "controller config")
    evidence_request = _read_json(args.evidence_request, "evidence request")
    config = controller_config_from_mapping(repository_root, config_value)

    raw_key = os.environ.get(RUNNER_HMAC_ENV)
    if raw_key is None:
        raise RuntimeError(f"{RUNNER_HMAC_ENV} is required")
    try:
        key = bytes.fromhex(raw_key)
    except ValueError as exc:
        raise RuntimeError(f"{RUNNER_HMAC_ENV} must be hexadecimal") from exc
    if len(key) < 32:
        raise RuntimeError(f"{RUNNER_HMAC_ENV} must contain at least 32 bytes")

    room_root = (
        canonical_paper_ai_rooms_root(repository_root)
        / config.experiment_id
        / config.room_id
    ).resolve(strict=True)
    broker_ledger = room_root / BROKER_LEDGER_NAME
    if not broker_ledger.is_file() or broker_ledger.is_symlink():
        raise RuntimeError("canonical broker ledger is unavailable")
    runner = DojoAIInventoryRunnerClient(derive_broker_socket_path(broker_ledger), key)
    result = run_ai_inventory_cycle(config, evidence_request, runner)
    output: dict[str, Any] = {
        "contract": "QR_DOJO_AI_INVENTORY_CONTROLLER_RESULT_V1",
        "cycle_sha256": result.cycle_record["cycle_sha256"],
        "decision_sha256": result.decision["decision_sha256"],
        "action": result.decision["action"],
        "applied_receipt_sha256": result.applied_receipt["applied_receipt_sha256"],
        "admission_reference": result.admission_reference,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    sys.stdout.write(
        json.dumps(
            output,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return 0


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_absolute():
        raise RuntimeError(f"{label} path must be absolute")
    try:
        item_stat = path.lstat()
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"{label} cannot be read") from exc
    if path.is_symlink() or not path.is_file() or item_stat.st_size != len(raw):
        raise RuntimeError(f"{label} must be one stable regular file")
    if not raw or len(raw) > MAX_INPUT_BYTES:
        raise RuntimeError(f"{label} size is invalid")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
