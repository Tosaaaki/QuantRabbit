"""Separated OANDA LIVE zero-order launchd services."""
from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from oanda_live_feed import OandaLiveRecorder, load_approved_live_credentials
from shadow_runtime import (
    HashLedger,
    RuntimeLock,
    atomic_json,
    canonical_bytes,
    canonical_hash,
    parse_utc,
    real_dir,
    secure_read,
    sha256_file,
    utc_text,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
SERVICE_ROOT = PACKAGE_ROOT / "runs" / "oanda_live_launchd_v1"
SOURCE_COMMIT = "907195888ae5671d18084d59a4458dc70f3df7c8"
SHARED_RUNTIME_HASH = canonical_hash({
    "source_commit": SOURCE_COMMIT,
    "oanda_contract_sha256": sha256_file(PACKAGE_ROOT / "oanda_live_runtime_contract.json"),
    "topology": "OANDA_LIVE_GET_ONLY_ZERO_ORDER_V1",
})
RUNTIME_SOURCE_PATHS = (
    PACKAGE_ROOT / "oanda_launchd_runtime.py",
    PACKAGE_ROOT / "oanda_live_feed.py",
    PACKAGE_ROOT / "shadow_runtime.py",
    PACKAGE_ROOT / "oanda_live_runtime_contract.json",
    PACKAGE_ROOT / "oanda_launchagents" / "com.quantrabbit.oanda-live.feed-recorder.plist",
    PACKAGE_ROOT / "oanda_launchagents" / "com.quantrabbit.oanda-live.bot-shadow.plist",
    PACKAGE_ROOT / "oanda_launchagents" / "com.quantrabbit.oanda-live.llm-inventory.plist",
    PACKAGE_ROOT / "oanda_launchagents" / "com.quantrabbit.oanda-live.watchdog.plist",
)


def runtime_source_hashes() -> dict[str, str]:
    return {str(path.relative_to(PACKAGE_ROOT)): sha256_file(path) for path in RUNTIME_SOURCE_PATHS}


RUNTIME_SOURCE_HASHES = runtime_source_hashes()
SERVICE_ATTESTATION_HASH = canonical_hash({
    "candidate_runtime_hash": SHARED_RUNTIME_HASH,
    "runtime_source_sha256": RUNTIME_SOURCE_HASHES,
})
LABELS = {
    "feed": "com.quantrabbit.oanda-live.feed-recorder",
    "bot": "com.quantrabbit.oanda-live.bot-shadow",
    "llm": "com.quantrabbit.oanda-live.llm-inventory",
    "watchdog": "com.quantrabbit.oanda-live.watchdog",
}
LLM_MODEL = "gpt-5.6-sol"
LLM_REASONING = "high"
CODEX_BIN = "/Users/tossaki/.local/bin/codex"
ALLOWED_ACTIONS = ("ADD", "FREEZE", "UNWIND", "RESET")
STOP = False


def request_stop(*_: object) -> None:
    global STOP
    STOP = True


def service_status(counters: dict[str, int], run_state: str = "RUNNING", blocked: bool = False) -> dict[str, Any]:
    return {
        "run_state": run_state,
        "feed_blocked": blocked,
        "runtime_hash": SHARED_RUNTIME_HASH,
        "counters": counters,
    }


def run_feed(max_seconds: float) -> int:
    account_id, token = load_approved_live_credentials()
    root = SERVICE_ROOT / "feed"
    recorder = OandaLiveRecorder(root)
    recorder.mark_approved_credential_file_read()
    status = recorder.run_live(
        account_id,
        token,
        max_seconds,
        runtime_hash=SERVICE_ATTESTATION_HASH,
    )
    return 2 if status["feed_blocked"] else 0


def _completed_bars(raw: HashLedger) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    max_bucket: dict[str, int] = {}
    for row in raw.rows:
        event = {**row["payload"], "raw_record_hash": row["record_hash"]}
        stamp = parse_utc(event["event_time_utc"])
        bucket = int(stamp.timestamp()) // 300 * 300
        key = (event["instrument"], bucket)
        grouped.setdefault(key, []).append(event)
        max_bucket[event["instrument"]] = max(bucket, max_bucket.get(event["instrument"], bucket))
    bars = []
    for (instrument, bucket), events in sorted(grouped.items()):
        if bucket >= max_bucket[instrument]:
            continue
        events.sort(key=lambda x: (x["event_time_utc"], x["arrival_time_utc"], x["event_id"]))
        bars.append({
            "schema_version": 1,
            "runtime_hash": SHARED_RUNTIME_HASH,
            "instrument": instrument,
            "timeframe": "M5",
            "start_utc": utc_text(datetime.fromtimestamp(bucket, timezone.utc)),
            "end_utc": utc_text(datetime.fromtimestamp(bucket + 300, timezone.utc)),
            "period_state": "SEALED",
            "event_count": len(events),
            "first_source_time_utc": events[0]["event_time_utc"],
            "last_source_time_utc": events[-1]["event_time_utc"],
            "first_arrival_time_utc": events[0]["arrival_time_utc"],
            "arrival_watermark_utc": max(x["arrival_time_utc"] for x in events),
            "bid_o": events[0]["bid"],
            "bid_h": max(x["bid"] for x in events),
            "bid_l": min(x["bid"] for x in events),
            "bid_c": events[-1]["bid"],
            "ask_o": events[0]["ask"],
            "ask_h": max(x["ask"] for x in events),
            "ask_l": min(x["ask"] for x in events),
            "ask_c": events[-1]["ask"],
            "raw_record_set_hash": canonical_hash([x["raw_record_hash"] for x in events]),
            "strategy_status": "RESEARCH_NOT_ADMITTED",
            "natural_r5_proposal": False,
            "external_orders": 0,
        })
    return bars


def bot_process_once() -> dict[str, int]:
    feed_raw = HashLedger(SERVICE_ROOT / "feed" / "ledgers" / "raw_bbo.jsonl")
    root = SERVICE_ROOT / "bot"
    real_dir(root)
    real_dir(root / "ledgers")
    completed = HashLedger(root / "ledgers" / "completed_m5.jsonl")
    control = HashLedger(root / "ledgers" / "control.jsonl")
    for bar in _completed_bars(feed_raw):
        planned: list[dict[str, Any]] = []
        record_id = f"m5::{bar['instrument']}::{bar['start_utc']}"
        row = completed.plan(bar, record_id, planned)
        completed.append_rows(planned)
        control_planned: list[dict[str, Any]] = []
        control.plan({
            "event": "R5_NATURAL_PROPOSAL_NOT_EMITTED",
            "bar_record_hash": row["record_hash"],
            "candidate_scope": "ACCOUNTING_ONLY_NOT_CAUSAL_SIGNAL_ADMISSION",
            "external_orders": 0,
        }, f"no-proposal::{row['record_hash']}", control_planned)
        control.append_rows(control_planned)
    return {
        "market_events": len(feed_raw.rows),
        "completed_m5": len(completed.rows),
        "natural_r5_proposals": 0,
        "virtual_fills": 0,
        "llm_calls": 0,
        "external_order_attempts": 0,
        "external_orders": 0,
    }


def run_bot(max_seconds: float) -> int:
    global STOP
    STOP = False
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    root = SERVICE_ROOT / "bot"
    deadline = time.monotonic() + max_seconds
    with RuntimeLock(root, SERVICE_ATTESTATION_HASH) as lock:
        while not STOP and time.monotonic() < deadline:
            counters = bot_process_once()
            lock.heartbeat(service_status(counters))
            time.sleep(2.0)
        counters = bot_process_once()
        lock.heartbeat(service_status(counters, "STOPPED_GRACEFULLY"))
    return 0


def llm_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ALLOWED_ACTIONS)},
            "currency_cap": {"type": "integer", "minimum": 0, "maximum": 1000000000},
            "mode": {"type": "string", "enum": ["SHADOW_ONLY"]},
            "valid_until": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string", "maxLength": 240},
        },
        "required": ["action", "currency_cap", "mode", "valid_until", "confidence", "reason"],
        "additionalProperties": False,
    }


def actual_model(prompt: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="qr-oanda-llm-") as tmp:
        schema = Path(tmp) / "schema.json"
        output = Path(tmp) / "output.json"
        schema.write_bytes(canonical_bytes(llm_output_schema()))
        subprocess.run([
            CODEX_BIN, "exec", "-", "--model", LLM_MODEL,
            "-c", 'model_reasoning_effort="high"', "--ephemeral",
            "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
            "--sandbox", "read-only", "--output-schema", str(schema),
            "--output-last-message", str(output),
        ], input=prompt, text=True, capture_output=True, check=True, timeout=180, cwd=tmp)
        return json.loads(output.read_text(encoding="utf-8"))


def process_llm_trigger(runner: Callable[[str], dict[str, Any]] = actual_model) -> dict[str, int]:
    root = SERVICE_ROOT / "llm"
    trigger_path = SERVICE_ROOT / "triggers" / "llm_inventory_request.json"
    real_dir(root)
    real_dir(root / "ledgers")
    receipts = HashLedger(root / "ledgers" / "receipts.jsonl")
    if not trigger_path.exists():
        return {"triggers": 0, "llm_calls": len(receipts.rows), "external_order_attempts": 0, "external_orders": 0}
    trigger = json.loads(secure_read(trigger_path).decode("utf-8", "strict"))
    required = {"trigger_id", "runtime_hash", "inventory_snapshot_hash", "open_inventory_count", "created_at_utc", "evidence_eligible", "profit_evidence", "external_orders"}
    if set(trigger) != required or trigger["runtime_hash"] != SHARED_RUNTIME_HASH:
        raise RuntimeError("LLM_TRIGGER_SCHEMA_MISMATCH")
    if trigger["evidence_eligible"] or trigger["profit_evidence"] or trigger["external_orders"] != 0:
        raise RuntimeError("LLM_TRIGGER_AUTHORITY_MISMATCH")
    record_id = f"llm::{trigger['trigger_id']}"
    if any(row["record_id"] == record_id for row in receipts.rows):
        return {"triggers": 1, "llm_calls": len(receipts.rows), "external_order_attempts": 0, "external_orders": 0}
    request_time = datetime.now(timezone.utc)
    prompt = (
        "Return one JSON inventory decision. Allowed action: ADD/FREEZE/UNWIND/RESET. "
        "mode=SHADOW_ONLY; currency_cap=0..1000000000 JPY microunits; valid_until within 2h. "
        "Do not control order, direction, fill, TP, SL, leverage, cost, or hard guard. External orders=0. "
        f"request_time={utc_text(request_time)} snapshot={canonical_bytes(trigger).decode()}"
    )
    output = runner(prompt)
    if set(output) != {"action", "currency_cap", "mode", "valid_until", "confidence", "reason"}:
        raise RuntimeError("LLM_OUTPUT_SCHEMA_MISMATCH")
    if output["action"] not in ALLOWED_ACTIONS or output["mode"] != "SHADOW_ONLY":
        raise RuntimeError("LLM_OUTPUT_AUTHORITY_MISMATCH")
    if type(output["currency_cap"]) is not int or not 0 <= output["currency_cap"] <= 1000000000:
        raise RuntimeError("LLM_OUTPUT_CAP_MISMATCH")
    if not request_time < parse_utc(output["valid_until"]) <= request_time + timedelta(hours=2):
        raise RuntimeError("LLM_OUTPUT_EXPIRY_MISMATCH")
    decision_time = datetime.now(timezone.utc)
    payload = {
        "kind": "ACTUAL_LLM_INVENTORY_RECEIPT",
        "runtime_hash": SHARED_RUNTIME_HASH,
        "service_attestation_hash": SERVICE_ATTESTATION_HASH,
        "model": LLM_MODEL,
        "reasoning": LLM_REASONING,
        "prompt_full": prompt,
        "prompt_sha256": canonical_hash(prompt),
        "input": trigger,
        "input_sha256": canonical_hash(trigger),
        "output": output,
        "output_sha256": canonical_hash(output),
        "decision_timestamp_utc": utc_text(decision_time),
        "arrival_timestamp_utc": utc_text(datetime.now(timezone.utc)),
        "individual_order_control": False,
        "hard_guard_mutation": False,
        "external_order_attempts": 0,
        "external_orders": 0,
    }
    planned: list[dict[str, Any]] = []
    receipts.plan(payload, record_id, planned)
    receipts.append_rows(planned)
    return {"triggers": 1, "llm_calls": len(receipts.rows), "external_order_attempts": 0, "external_orders": 0}


def run_llm() -> int:
    root = SERVICE_ROOT / "llm"
    with RuntimeLock(root, SERVICE_ATTESTATION_HASH) as lock:
        counters = process_llm_trigger()
        lock.heartbeat(service_status(counters, "IDLE_TRIGGER_DRIVEN"))
    return 0


def _verify_heartbeat(path: Path, max_age: float) -> dict[str, Any]:
    row = json.loads(secure_read(path).decode("utf-8", "strict"))
    unsigned = {k: v for k, v in row.items() if k != "heartbeat_hash"}
    if canonical_hash(unsigned) != row.get("heartbeat_hash"):
        raise RuntimeError("CORRUPT_HEARTBEAT")
    if row.get("strategy_hash") != SERVICE_ATTESTATION_HASH:
        raise RuntimeError("RUNTIME_HASH_MISMATCH")
    if (datetime.now(timezone.utc) - parse_utc(row["beat_at_utc"])).total_seconds() > max_age:
        raise RuntimeError("STALE_HEARTBEAT")
    if row["counters"].get("external_orders") != 0 or row["counters"].get("external_order_attempts") != 0:
        raise RuntimeError("ORDER_AUTHORITY_BREACH")
    return row


def run_watchdog() -> int:
    root = SERVICE_ROOT / "watchdog"
    real_dir(root)
    feed = _verify_heartbeat(SERVICE_ROOT / "feed" / "heartbeat.json", 45.0)
    bot = _verify_heartbeat(SERVICE_ROOT / "bot" / "heartbeat.json", 45.0)
    counters = {
        "feed_events": int(feed["counters"].get("market_events_accepted", 0)),
        "bot_bars": int(bot["counters"].get("completed_m5", 0)),
        "llm_calls": 0,
        "external_order_attempts": 0,
        "external_orders": 0,
    }
    with RuntimeLock(root, SERVICE_ATTESTATION_HASH) as lock:
        lock.heartbeat(service_status(counters, "HEALTHY"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=("feed", "bot", "llm", "watchdog"))
    parser.add_argument("--seconds", type=float, default=86400.0)
    args = parser.parse_args(argv)
    real_dir(SERVICE_ROOT)
    real_dir(SERVICE_ROOT / "triggers")
    real_dir(SERVICE_ROOT / "logs")
    if args.service == "feed":
        return run_feed(args.seconds)
    if args.service == "bot":
        return run_bot(args.seconds)
    if args.service == "llm":
        return run_llm()
    return run_watchdog()


if __name__ == "__main__":
    raise SystemExit(main())
